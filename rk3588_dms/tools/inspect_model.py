#!/usr/bin/env python3
"""ONNX 模型结构检查工具（RKNN 迁移专用）。

用途:
  在转换 RKNN 之前, 检查一个 ONNX 检测模型的结构是否适合直接转换,
  输出 opset / 输入输出 shape / 算子统计 / 是否包含 NMS、Sigmoid、
  DFL、动态维度等对 RKNN 转换或后处理有影响的信息。

用法:
  python rk3588_dms/tools/inspect_model.py path/to/model.onnx [more.onnx ...]
  python rk3588_dms/tools/inspect_model.py model.onnx --json

依赖:
  pip install onnx        (只读图结构, 不需要 onnxruntime / rknn-toolkit2)

退出码:
  0 正常; 1 文件不存在; 2 依赖缺失; 3 模型解析失败
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# RKNN-Toolkit2 常见不支持/需谨慎的算子(以 rknn-toolkit2 支持列表为准, 逐步更新)
RKNN_CAUTION_OPS = {
    "NonMaxSuppression": "RKNN 不支持内置 NMS, 必须在模型外做后处理",
    "EfficientNMS": "TensorRT 专用 NMS 节点, RKNN 不支持",
    "TopK": "部分版本支持有限, 若在 Detect 头内需删除",
    "Split": "通常支持, 但动态 split 需要注意",
    "If": "控制流算子, RKNN 不支持",
    "Loop": "控制流算子, RKNN 不支持",
    "Gemm": "支持但需注意转置融合",
}

# YOLOv8/v11 检测头常见结构特征
DFL_SIGN_OPS = ("Conv", "Reshape", "Transpose", "Slice", "Concat", "Mul", "Add", "Sigmoid", "Softmax")


def _dim_to_str(dim) -> str:
    if dim.HasField("dim_param") and dim.dim_param:
        return dim.dim_param
    if dim.HasField("dim_value"):
        return str(dim.dim_value)
    return "?"


def _tensor_shape(value_info) -> dict:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type) if tensor_type.elem_type else "?",
        "shape": [_dim_to_str(d) for d in tensor_type.shape.dim],
        "dynamic": any(
            (d.HasField("dim_param") and d.dim_param) or not (d.HasField("dim_value") or (d.HasField("dim_param") and d.dim_param))
            for d in tensor_type.shape.dim
        ),
    }


def _traverse_up(graph, node_name: str):
    return graph.node[node_name]


# ---------------------------------------------------------------------------
# 零依赖回退: 手写最小 protobuf 解析(无 onnx 包时仍可读取图结构)。
# 只提取 inspect 所需字段: opset / producer / 节点 op_type / 输入输出 shape /
# initializer 参数量。字段号参考 onnx.proto3。
# ---------------------------------------------------------------------------
_ONNX_ELEM_TYPE = {
    0: "undefined", 1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64", 12: "uint32",
    13: "uint64", 16: "bfloat16",
}


def _read_varint(buf: bytes, pos: int):
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7


def _iter_fields(buf: bytes):
    pos = 0
    end = len(buf)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire == 5:
            value = buf[pos : pos + 4]
            pos += 4
        elif wire == 1:
            value = buf[pos : pos + 8]
            pos += 8
        else:
            raise ValueError(f"不支持的 wire type {wire}")
        yield field, wire, value


def _parse_type_tensor(buf: bytes) -> dict:
    elem_type, dims = "?", []
    for field, _wire, value in _iter_fields(buf):
        if field == 1:
            elem_type = _ONNX_ELEM_TYPE.get(value, str(value))
        elif field == 2:  # TensorShapeProto
            for dfield, _dwire, dvalue in _iter_fields(value):
                if dfield == 1:  # Dimension
                    dim_value, dim_param = None, None
                    for f2, _w2, v2 in _iter_fields(dvalue):
                        if f2 == 1:
                            dim_value = v2
                        elif f2 == 2:
                            dim_param = v2.decode("utf-8", "replace")
                    dims.append(dim_param if dim_param else str(dim_value if dim_value is not None else "?"))
    return {"dtype": elem_type, "shape": dims, "dynamic": any(not d.isdigit() for d in dims)}


def _parse_value_info(buf: bytes) -> dict:
    name, tensor = "", {"dtype": "?", "shape": [], "dynamic": False}
    for field, _wire, value in _iter_fields(buf):
        if field == 1:
            name = value.decode("utf-8", "replace")
        elif field == 2:  # TypeProto
            for f2, _w2, v2 in _iter_fields(value):
                if f2 == 1:  # tensor_type
                    tensor = _parse_type_tensor(v2)
    return {"name": name, **tensor}


def _parse_graph_minimal(buf: bytes) -> dict:
    op_counts: dict[str, int] = {}
    inputs, outputs, inits = [], [], []
    for field, _wire, value in _iter_fields(buf):
        if field == 1:  # NodeProto
            for f2, _w2, v2 in _iter_fields(value):
                if f2 == 4:
                    op = v2.decode("utf-8", "replace")
                    op_counts[op] = op_counts.get(op, 0) + 1
        elif field == 11:
            inputs.append(_parse_value_info(value))
        elif field == 12:
            outputs.append(_parse_value_info(value))
        elif field == 5:  # TensorProto (initializer)
            init_dims = []
            for f2, w2, v2 in _iter_fields(value):
                if f2 == 1:
                    if w2 == 0:
                        init_dims.append(v2)
                    else:  # packed repeated int64
                        pos = 0
                        while pos < len(v2):
                            dim, pos = _read_varint(v2, pos)
                            init_dims.append(dim)
            count = 1
            for d in init_dims:
                count *= d
            inits.append(count)
    return {
        "op_counts": op_counts,
        "inputs": inputs,
        "outputs": outputs,
        "param_count": sum(inits),
        "initializer_count": len(inits),
    }


def _inspect_minimal(path: Path) -> dict:
    data = path.read_bytes()
    producer_name = producer_version = doc_string = ""
    opsets: dict[str, int] = {}
    metadata: dict[str, str] = {}
    graph_info = {"op_counts": {}, "inputs": [], "outputs": [], "param_count": 0, "initializer_count": 0}
    ir_version = None
    for field, _wire, value in _iter_fields(data):
        if field == 1:
            ir_version = value
        elif field == 2:
            producer_name = value.decode("utf-8", "replace")
        elif field == 3:
            producer_version = value.decode("utf-8", "replace")
        elif field == 6:
            doc_string = value.decode("utf-8", "replace")
        elif field == 7:
            graph_info = _parse_graph_minimal(value)
        elif field == 8:  # OperatorSetIdProto
            domain, version = "ai.onnx", 0
            for f2, _w2, v2 in _iter_fields(value):
                if f2 == 1:
                    domain = v2.decode("utf-8", "replace") or "ai.onnx"
                elif f2 == 2:
                    version = v2
            opsets[domain] = version
        elif field == 14:  # metadata StringStringEntryProto
            key = val = ""
            for f2, _w2, v2 in _iter_fields(value):
                if f2 == 1:
                    key = v2.decode("utf-8", "replace")
                elif f2 == 2:
                    val = v2.decode("utf-8", "replace")
            metadata[key] = val

    op_counts = graph_info["op_counts"]
    flags = {
        "contains_nms": any("NMS" in op or "NonMaxSuppression" in op for op in op_counts),
        "sigmoid_count": op_counts.get("Sigmoid", 0),
        "softmax_count": op_counts.get("Softmax", 0),
        "reshape_count": op_counts.get("Reshape", 0),
        "transpose_count": op_counts.get("Transpose", 0),
        "dfl_like_detect_head": op_counts.get("Reshape", 0) >= 2 and op_counts.get("Conv", 0) >= 20,
        "dynamic_input_shape": any(i["dynamic"] for i in graph_info["inputs"]),
        "dynamic_output_shape": any(o["dynamic"] for o in graph_info["outputs"]),
    }
    return {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "producer": {
            "producer_name": producer_name,
            "producer_version": producer_version,
            "ir_version": ir_version if ir_version is not None else "?",
            "opset": [f"{domain}:{version}" for domain, version in opsets.items()],
            "doc": doc_string,
        },
        "metadata": metadata,
        "inputs": graph_info["inputs"],
        "outputs": graph_info["outputs"],
        "output_notes": [],
        "node_count": sum(op_counts.values()),
        "op_counts": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
        "param_count": graph_info["param_count"],
        "flags": flags,
        "rknn_caution_ops": [(op, RKNN_CAUTION_OPS[op]) for op in sorted(op_counts) if op in RKNN_CAUTION_OPS],
        "parser": "minimal(未安装 onnx 包, 手写 protobuf 解析)",
    }


def inspect_model(path: Path) -> dict:
    try:
        import onnx  # noqa: F401
    except ImportError:
        return _inspect_minimal(path)

    model = onnx.load(str(path))
    graph = model.graph

    op_counts: dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    inputs = [_tensor_shape(v) for v in graph.input]
    outputs = [_tensor_shape(v) for v in graph.output]

    # 判断输出是否是 Ultralytics 默认 "已拼接+解析" 的 [1, 4+nc, N] 形式,
    # 以及浏览器 parseDetections 兼容性推断。
    output_notes = []
    for out in outputs:
        shape = out["shape"]
        if len(shape) == 3 and shape[0] == "1":
            a, n = (shape[1], shape[2]) if shape[1] != "?" and shape[2] != "?" else ("?", "?")
            output_notes.append(
                f"输出 {out['name']}: [1, {a}, {n}] -> "
                + ("attr-major(每行一个属性, Ultralytics 默认导出)" if a not in ("?",) and n not in ("?",) and int(a) < int(n) else "candidate-major(每行一个候选框)")
            )
        elif len(shape) == 4:
            output_notes.append(f"输出 {out['name']}: [1, C, H, W] 特征图形式(常见于 RKNN Model Zoo 风格导出, 后处理需自行 DFL 解码)")

    cautions = [(op, RKNN_CAUTION_OPS[op]) for op in sorted(op_counts) if op in RKNN_CAUTION_OPS]
    has_nms = any("NMS" in op or "NonMaxSuppression" in op for op in op_counts)
    sigmoid_count = op_counts.get("Sigmoid", 0)
    reshape_count = op_counts.get("Reshape", 0)
    transpose_count = op_counts.get("Transpose", 0)
    softmax_count = op_counts.get("Softmax", 0)

    # DFL / Detect 头启发式判断: 尾部大量 Conv+Reshape+Concat
    tail = graph.node[-40:]
    tail_ops = [n.op_type for n in tail]
    dfl_like = sum(1 for op in tail_ops if op in DFL_SIGN_OPS) >= 12 and reshape_count >= 2

    metadata = {p.key: p.value for p in model.metadata_props}
    producer = {
        "producer_name": model.producer_name or "",
        "producer_version": model.producer_version or "",
        "ir_version": model.ir_version,
        "opset": [f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import],
        "doc": (model.graph.doc_string or "").strip(),
    }

    param_count = 0
    for init in graph.initializer:
        size = 1
        for d in init.dims:
            size *= d
        param_count += size

    return {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "producer": producer,
        "metadata": metadata,
        "inputs": inputs,
        "outputs": outputs,
        "output_notes": output_notes,
        "node_count": len(graph.node),
        "op_counts": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
        "param_count": param_count,
        "flags": {
            "contains_nms": has_nms,
            "sigmoid_count": sigmoid_count,
            "softmax_count": softmax_count,
            "reshape_count": reshape_count,
            "transpose_count": transpose_count,
            "dfl_like_detect_head": dfl_like,
            "dynamic_input_shape": any(i["dynamic"] for i in inputs),
            "dynamic_output_shape": any(o["dynamic"] for o in outputs),
        },
        "rknn_caution_ops": cautions,
    }


def print_report(report: dict) -> None:
    p = report["producer"]
    print("=" * 72)
    print(f"模型文件 : {report['file']}")
    print(f"文件大小 : {report['size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"生成器   : {p['producer_name']} {p['producer_version']}")
    print(f"IR 版本  : {p['ir_version']}    Opset: {', '.join(p['opset'])}")
    if p["doc"]:
        print(f"说明     : {p['doc'][:120]}")
    if report["metadata"]:
        for key, value in report["metadata"].items():
            print(f"meta     : {key} = {value[:160]}")
    print("-" * 72)
    for item in report["inputs"]:
        dyn = " [动态维度!]" if item["dynamic"] else ""
        print(f"输入     : {item['name']}  shape={item['shape']}  dtype={item['dtype']}{dyn}")
    for item in report["outputs"]:
        dyn = " [动态维度!]" if item["dynamic"] else ""
        print(f"输出     : {item['name']}  shape={item['shape']}  dtype={item['dtype']}{dyn}")
    for note in report["output_notes"]:
        print(f"  * {note}")
    print("-" * 72)
    print(f"节点总数 : {report['node_count']}    参数量: {report['param_count']:,}")
    top_ops = list(report["op_counts"].items())[:12]
    print("算子统计 : " + ", ".join(f"{op}×{count}" for op, count in top_ops))
    print("-" * 72)
    flags = report["flags"]
    print("迁移特征 :")
    print(f"  内置 NMS          : {'是 —— RKNN 需删除或模型外处理!' if flags['contains_nms'] else '否'}")
    print(f"  Sigmoid 数量      : {flags['sigmoid_count']} (Detect 头通常应有, 类别分数)")
    print(f"  Softmax 数量      : {flags['softmax_count']} (DFL 分布回归特征)")
    print(f"  Reshape/Transpose : {flags['reshape_count']}/{flags['transpose_count']} (输出整形成 [1,4+nc,N] 的痕迹)")
    print(f"  DFL 风格 Detect 头: {'是' if flags['dfl_like_detect_head'] else '否/不确定'}")
    print(f"  动态输入 shape    : {'是 —— 转换前需固定!' if flags['dynamic_input_shape'] else '否'}")
    if report["rknn_caution_ops"]:
        print("  注意算子:")
        for op, reason in report["rknn_caution_ops"]:
            print(f"    - {op}: {reason}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ONNX 模型结构检查(RKNN 迁移预检)")
    parser.add_argument("models", nargs="+", help="ONNX 模型路径")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非人类可读文本")
    args = parser.parse_args(argv)

    reports = []
    for raw in args.models:
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"[FAIL] 文件不存在: {path}", file=sys.stderr)
            return 1
        try:
            reports.append(inspect_model(path))
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] 解析 {path} 失败: {exc}", file=sys.stderr)
            return 3

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
