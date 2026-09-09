/*
 * Annotation Replay Player — 蓝/红/黄标注循环放映(独立组件,零依赖)
 *
 * 用法:
 *   <div id="player" style="width:100%; aspect-ratio:16/9;"></div>
 *   <script src="replay-player.js"></script>
 *   <script>
 *     const player = mountAnnotationReplay(document.getElementById("player"), {
 *       durationMs: 5600,
 *       frames: [ /* 录制 JSON 的 frames 数组,坐标为 0~1 归一化 * / ]
 *     }, { speed: 1, loop: true });
 *     // player.pause(); player.play(); player.setSpeed(2); player.seek(1000); player.destroy();
 *   </script>
 *
 * 数据格式(与检测端 storage/annotation-tracks/*.json 一致):
 *   { durationMs, frames: [{ t, aw, ah, faceBox, paths, dots, gaze, redBoxes, poseLabel, alarm }] }
 */
(function (global) {
  "use strict";

  const ALARM_LABELS = {
    phone: "Phone Call 打电话报警",
    drowsy: "Eye Closure 闭眼疲劳报警",
    head_down: "Head Down 低头报警",
    gaze_off: "Gaze Offset 视线偏离报警",
  };
  const SPEEDS = [0.5, 1, 2, 4, 8];

  global.mountAnnotationReplay = function (container, payload, options) {
    const opts = Object.assign({ speed: 1, loop: true, background: "#0b1220", showAlarmBanner: true }, options || {});
    if (!container) throw new Error("mountAnnotationReplay: 缺少容器元素");
    if (!payload || !Array.isArray(payload.frames) || !payload.frames.length) {
      throw new Error("mountAnnotationReplay: 数据需要包含非空 frames 数组");
    }

    const frames = payload.frames;
    const durationMs = payload.durationMs || frames[frames.length - 1].t - frames[0].t || 0;

    const canvas = document.createElement("canvas");
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    container.appendChild(canvas);
    const context = canvas.getContext("2d");

    let playing = true;
    let speed = Number(opts.speed) || 1;
    let playheadMs = 0;
    let lastTickAt = performance.now();
    let destroyed = false;
    let rafId = 0;

    function resize() {
      const rect = container.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(rect.width * dpr));
      const h = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    }

    function frameAt(playhead) {
      const time = frames[0].t + Math.min(Math.max(playhead, 0), durationMs);
      let low = 0;
      let high = frames.length - 1;
      while (low < high) {
        const mid = (low + high + 1) >> 1;
        if (frames[mid].t <= time) low = mid;
        else high = mid - 1;
      }
      return frames[low];
    }

    function drawFrame(frame, now) {
      const w = canvas.width;
      const h = canvas.height;
      context.clearRect(0, 0, w, h);
      context.fillStyle = opts.background;
      context.fillRect(0, 0, w, h);
      if (!frame) return;
      const x = (value) => value * w;
      const y = (value) => value * h;
      const lineWidth = Math.max(2, w / 420);

      // 蓝色:关键点连线(脸轮廓/眼睛/鼻梁/嘴唇)
      context.strokeStyle = "rgba(96, 165, 250, 0.7)";
      context.lineWidth = Math.max(1.2, lineWidth * 0.38);
      context.lineJoin = "round";
      (frame.paths || []).forEach((path) => {
        context.beginPath();
        path.forEach((point, index) => {
          if (index === 0) context.moveTo(x(point[0]), y(point[1]));
          else context.lineTo(x(point[0]), y(point[1]));
        });
        context.closePath();
        context.stroke();
      });

      // 蓝色:关键点圆点
      (frame.dots || []).forEach((dot) => {
        const key = dot[2] === 1;
        const radius = key ? lineWidth * 1.6 : lineWidth * 1.1;
        context.beginPath();
        context.fillStyle = key ? "#dbeafe" : "#3b82f6";
        context.arc(x(dot[0]), y(dot[1]), radius, 0, Math.PI * 2);
        context.fill();
        context.beginPath();
        context.strokeStyle = "rgba(248, 250, 252, 0.95)";
        context.lineWidth = Math.max(0.9, lineWidth * 0.22);
        context.arc(x(dot[0]), y(dot[1]), radius + 0.9, 0, Math.PI * 2);
        context.stroke();
      });

      // 黄色:视线箭头
      (frame.gaze || []).forEach((ray) => {
        const ox = x(ray.o[0]);
        const oy = y(ray.o[1]);
        const originRadius = Math.max(4.5, lineWidth * 1.55);
        context.save();
        context.lineCap = "round";
        context.shadowColor = "#f4f07a";
        context.shadowBlur = 10;
        context.fillStyle = "#fffbd1";
        context.beginPath();
        context.arc(ox, oy, originRadius, 0, Math.PI * 2);
        context.fill();
        if (ray.e) {
          const ex = x(ray.e[0]);
          const ey = y(ray.e[1]);
          const angle = Math.atan2(ey - oy, ex - ox);
          const arrowSize = Math.max(11, lineWidth * 3.6);
          context.strokeStyle = "rgba(244, 240, 122, 0.44)";
          context.lineWidth = Math.max(9, lineWidth * 2.7);
          context.shadowBlur = 24;
          context.beginPath();
          context.moveTo(ox, oy);
          context.lineTo(ex, ey);
          context.stroke();
          context.strokeStyle = "#fffbd1";
          context.lineWidth = Math.max(3.2, lineWidth * 0.92);
          context.shadowBlur = 12;
          context.beginPath();
          context.moveTo(ox, oy);
          context.lineTo(ex, ey);
          context.stroke();
          context.fillStyle = "#f4f07a";
          context.beginPath();
          context.moveTo(ex, ey);
          context.lineTo(ex - arrowSize * Math.cos(angle - Math.PI / 6), ey - arrowSize * Math.sin(angle - Math.PI / 6));
          context.lineTo(ex - arrowSize * Math.cos(angle + Math.PI / 6), ey - arrowSize * Math.sin(angle + Math.PI / 6));
          context.closePath();
          context.fill();
        } else {
          context.strokeStyle = "rgba(244, 240, 122, 0.84)";
          context.lineWidth = Math.max(2.2, lineWidth * 0.62);
          context.beginPath();
          context.arc(ox, oy, originRadius * 1.85, 0, Math.PI * 2);
          context.stroke();
        }
        context.restore();
      });

      // 蓝色:人脸检测框(四角括号)+ FACE 01 标签
      if (frame.faceBox) {
        const box = frame.faceBox;
        const bx1 = x(box.x1);
        const by1 = y(box.y1);
        const bx2 = x(box.x2);
        const by2 = y(box.y2);
        const bw = Math.max(1, bx2 - bx1);
        const bh = Math.max(1, by2 - by1);
        const corner = Math.max(18, Math.min(bw, bh) * 0.16);
        const strokeWidth = Math.max(2.4, lineWidth * 0.78);
        context.save();
        context.lineCap = "round";
        context.lineJoin = "round";
        context.shadowColor = "rgba(96, 165, 250, 0.75)";
        context.shadowBlur = 14;
        context.strokeStyle = "rgba(96, 165, 250, 0.32)";
        context.lineWidth = Math.max(1.2, strokeWidth * 0.48);
        context.strokeRect(bx1, by1, bw, bh);
        context.strokeStyle = "#70d7ff";
        context.lineWidth = strokeWidth;
        context.beginPath();
        context.moveTo(bx1, by1 + corner);
        context.lineTo(bx1, by1);
        context.lineTo(bx1 + corner, by1);
        context.moveTo(bx2 - corner, by1);
        context.lineTo(bx2, by1);
        context.lineTo(bx2, by1 + corner);
        context.moveTo(bx2, by2 - corner);
        context.lineTo(bx2, by2);
        context.lineTo(bx2 - corner, by2);
        context.moveTo(bx1 + corner, by2);
        context.lineTo(bx1, by2);
        context.lineTo(bx1, by2 - corner);
        context.stroke();
        context.shadowBlur = 0;
        context.font = `800 ${Math.max(11, w / 82)}px system-ui, sans-serif`;
        const labelWidth = context.measureText(box.label).width + 16;
        const labelHeight = Math.max(20, w / 60);
        const labelY = Math.max(labelHeight + 6, by1 - 6);
        context.fillStyle = "rgba(8, 16, 48, 0.76)";
        context.fillRect(bx1, labelY - labelHeight, labelWidth, labelHeight);
        context.fillStyle = "#dbeafe";
        context.fillText(box.label, bx1 + 8, labelY - 6);
        context.restore();
      }

      // 红色:行为框(Phone Call …%)
      (frame.redBoxes || []).forEach((box) => {
        const bx1 = x(box.x1);
        const by1 = y(box.y1);
        const bx2 = x(box.x2);
        const by2 = y(box.y2);
        const bw = Math.max(1, bx2 - bx1);
        const bh = Math.max(1, by2 - by1);
        const corner = Math.max(12, Math.min(bw, bh) * 0.16);
        context.save();
        context.lineJoin = "round";
        context.lineCap = "round";
        context.strokeStyle = "#ff4651";
        context.fillStyle = "rgba(255, 70, 81, 0.14)";
        context.lineWidth = lineWidth;
        context.shadowColor = "rgba(255, 70, 81, 0.82)";
        context.shadowBlur = 16;
        context.strokeRect(bx1, by1, bw, bh);
        context.shadowBlur = 0;
        context.fillRect(bx1, by1, bw, bh);
        context.font = `800 ${Math.max(13, w / 86)}px system-ui, sans-serif`;
        const labelWidth = context.measureText(box.label).width + 16;
        const labelHeight = Math.max(22, Math.max(13, w / 86) * 1.65);
        const labelX = Math.max(4, Math.min(bx1, w - labelWidth - 4));
        const labelY = by1 - labelHeight > 2 ? by1 - labelHeight : by1 + 2;
        context.fillStyle = "rgba(104, 21, 39, 0.88)";
        context.fillRect(labelX, labelY, labelWidth, labelHeight);
        context.fillStyle = "#ffe4e6";
        context.fillText(box.label, labelX + 8, labelY + labelHeight - 7);
        context.strokeStyle = "#ffe4e6";
        context.lineWidth = Math.max(1.4, lineWidth * 0.48);
        context.beginPath();
        context.moveTo(bx1, by1 + corner);
        context.lineTo(bx1, by1);
        context.lineTo(bx1 + corner, by1);
        context.moveTo(bx2 - corner, by1);
        context.lineTo(bx2, by1);
        context.lineTo(bx2, by1 + corner);
        context.moveTo(bx2, by2 - corner);
        context.lineTo(bx2, by2);
        context.lineTo(bx2 - corner, by2);
        context.moveTo(bx1 + corner, by2);
        context.lineTo(bx1, by2);
        context.lineTo(bx1, by2 - corner);
        context.stroke();
        context.fillStyle = "#ff4651";
        context.beginPath();
        context.arc(bx2, by1, Math.max(4, lineWidth * 1.8), 0, Math.PI * 2);
        context.fill();
        context.restore();
      });

      // 姿态角标签(P..° Y..° R..°)
      if (frame.poseLabel) {
        context.save();
        context.font = `700 ${Math.max(12, w / 72)}px system-ui, sans-serif`;
        const metrics = context.measureText(frame.poseLabel);
        const labelHeight = Math.max(22, w / 52);
        const labelX = Math.max(8, (w - metrics.width - 16) / 2);
        const labelY = h - 8;
        context.fillStyle = "rgba(8, 22, 39, 0.78)";
        context.fillRect(labelX, labelY - labelHeight, metrics.width + 16, labelHeight);
        context.fillStyle = "#dbeafe";
        context.fillText(frame.poseLabel, labelX + 8, labelY - 7);
        context.restore();
      }

      // 红色报警横幅(带脉冲呼吸)
      const alarms = frame.alarm || [];
      if (opts.showAlarmBanner && alarms.length) {
        const label = alarms.map((key) => "⚠ " + (ALARM_LABELS[key] || key)).join("　");
        const pulse = 0.62 + 0.38 * Math.abs(Math.sin(now / 300));
        context.save();
        context.globalAlpha = pulse;
        context.font = `800 ${Math.max(14, w / 46)}px system-ui, sans-serif`;
        const textWidth = context.measureText(label).width;
        const padX = Math.max(14, w / 60);
        const bannerHeight = Math.max(30, w / 24);
        const bannerY = Math.max(10, h * 0.04);
        const bannerX = (w - textWidth - padX * 2) / 2;
        context.fillStyle = "rgba(255, 70, 81, 0.92)";
        context.shadowColor = "rgba(255, 70, 81, 0.6)";
        context.shadowBlur = 20;
        const radius = bannerHeight / 2;
        context.beginPath();
        context.moveTo(bannerX + radius, bannerY);
        context.arcTo(bannerX + textWidth + padX * 2, bannerY, bannerX + textWidth + padX * 2, bannerY + bannerHeight, radius);
        context.arcTo(bannerX + textWidth + padX * 2, bannerY + bannerHeight, bannerX, bannerY + bannerHeight, radius);
        context.arcTo(bannerX, bannerY + bannerHeight, bannerX, bannerY, radius);
        context.arcTo(bannerX, bannerY, bannerX + textWidth + padX * 2, bannerY, radius);
        context.closePath();
        context.fill();
        context.shadowBlur = 0;
        context.fillStyle = "#fff1f2";
        context.fillText(label, bannerX + padX, bannerY + bannerHeight - bannerHeight / 2 + Math.max(5, w / 140));
        context.restore();
      }
    }

    function tick(now) {
      if (destroyed) return;
      resize();
      if (playing && frames.length > 1 && durationMs > 0) {
        // 后台标签页 rAF 暂停时钳制 delta,避免恢复后播放位置跳变
        playheadMs += Math.min(now - lastTickAt, 250) * speed;
        if (playheadMs >= durationMs) {
          if (opts.loop) playheadMs %= durationMs;
          else {
            playheadMs = durationMs;
            playing = false;
          }
        }
      }
      lastTickAt = now;
      drawFrame(frameAt(playheadMs), now);
      rafId = requestAnimationFrame(tick);
    }

    const resizeObserver = typeof ResizeObserver === "function" ? new ResizeObserver(() => resize()) : null;
    if (resizeObserver) resizeObserver.observe(container);

    lastTickAt = performance.now();
    rafId = requestAnimationFrame(tick);

    return {
      play() { playing = true; lastTickAt = performance.now(); },
      pause() { playing = false; },
      setSpeed(value) { speed = SPEEDS.includes(value) ? value : Number(value) || 1; },
      seek(ms) { playheadMs = Math.min(Math.max(ms, 0), durationMs); },
      get playing() { return playing; },
      get durationMs() { return durationMs; },
      destroy() {
        destroyed = true;
        cancelAnimationFrame(rafId);
        if (resizeObserver) resizeObserver.disconnect();
        canvas.remove();
      },
    };
  };
})(typeof window !== "undefined" ? window : this);
