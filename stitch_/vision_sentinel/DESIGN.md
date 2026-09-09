---
name: Vision Sentinel
colors:
  surface: '#ffffff'
  surface-dim: '#d8dadc'
  surface-bright: '#f7f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f4f6'
  surface-container: '#eceef0'
  surface-container-high: '#e6e8ea'
  surface-container-highest: '#e0e3e5'
  on-surface: '#191c1e'
  on-surface-variant: '#3e4850'
  inverse-surface: '#2d3133'
  inverse-on-surface: '#eff1f3'
  outline: '#6e7881'
  outline-variant: '#bec8d2'
  surface-tint: '#006591'
  primary: '#006591'
  on-primary: '#ffffff'
  primary-container: '#0ea5e9'
  on-primary-container: '#003751'
  inverse-primary: '#89ceff'
  secondary: '#00668a'
  on-secondary: '#ffffff'
  secondary-container: '#40c2fd'
  on-secondary-container: '#004d6a'
  tertiary: '#8a5100'
  on-tertiary: '#ffffff'
  tertiary-container: '#de8712'
  on-tertiary-container: '#4d2b00'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#c9e6ff'
  primary-fixed-dim: '#89ceff'
  on-primary-fixed: '#001e2f'
  on-primary-fixed-variant: '#004c6e'
  secondary-fixed: '#c4e7ff'
  secondary-fixed-dim: '#7bd0ff'
  on-secondary-fixed: '#001e2c'
  on-secondary-fixed-variant: '#004c69'
  tertiary-fixed: '#ffdcbd'
  tertiary-fixed-dim: '#ffb86e'
  on-tertiary-fixed: '#2c1600'
  on-tertiary-fixed-variant: '#693c00'
  background: '#f8fafc'
  on-background: '#191c1e'
  surface-variant: '#e0e3e5'
  text-primary: '#082f49'
  text-secondary: '#475569'
  success: '#22c55e'
  warning: '#f59e0b'
  danger: '#ef4444'
  border-subtle: '#e2e8f0'
typography:
  display-xl:
    fontFamily: Geist
    fontSize: 56px
    fontWeight: '300'
    lineHeight: '1'
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.2'
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: '1.3'
  body-md:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  body-sm:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
  label-bold:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '700'
    lineHeight: '1'
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  container-margin: 2.5rem
  gutter: 1.5rem
  bento-padding: 2rem
  component-gap: 0.75rem
---

## Brand & Style

The design system is a sophisticated evolution of modern dashboard aesthetics, blending **Minimalism** with subtle **Glassmorphism**. It is designed to evoke a sense of clarity, precision, and high-fidelity performance, specifically tailored for professional environments that require high data density without the associated cognitive load.

The visual narrative is "Atmospheric Tech"—a combination of airy, open layouts and crisp, professional accents. It utilizes a "Bento Box" structure to organize information into distinct, hyper-rounded pods, creating a modular interface that feels both structured and approachable. The overall emotional response should be one of "calm control," achieved through expansive white space, soft blue transitions, and high-contrast typography.

## Colors

The palette is anchored by a high-contrast relationship between pure white surfaces and deep navy typography. 

- **Primary Identity:** Defined by a professional sky-blue gradient. Use `#0ea5e9` as the base and `#38bdf8` for highlights and hover states.
- **Background & Surfaces:** The interface uses a "layered white" approach. The base background is a cool off-white (`#f8fafc`), while all functional bento containers and cards are pure white (`#ffffff`).
- **Functional States:** Status colors (Success, Warning, Danger) should be implemented with soft, desaturated background tints for containers (e.g., 10% opacity) paired with high-saturation text or icons to maintain a professional "SaaS" look rather than a loud, consumer-grade palette.

## Typography

**Geist** is the exclusive typeface for this design system, chosen for its monolinear, technical precision. 

To maintain the "Bento" aesthetic, display typography and headlines should use tighter letter-spacing and heavier weights to anchor the containers. Large data points (Display XL) should use a lighter weight to prevent the UI from feeling heavy. For body text, prioritize readability with a generous 1.6x line height. Metadata and small labels should always be uppercase with increased tracking to differentiate them from interactive body elements.

## Layout & Spacing

The system follows a **Fluid Bento Grid** model. Content is organized into modular containers that span a 12-column system. 

- **Desktop:** Use a 2.5rem (40px) margin for the outer container with a 1.5rem (24px) gutter between bento tiles.
- **Bento Logic:** Tiles should be grouped by context. A single row might contain one large tile (8 columns) and one medium tile (4 columns). 
- **Reflow:** On tablet, the grid transitions to 8 columns. On mobile, all tiles stack into a single column with margins reduced to 1.25rem to maximize screen utility.

## Elevation & Depth

Depth is communicated through "Atmospheric Softness"—low-contrast layers that suggest height without creating visual noise.

- **The Bento Lift:** White cards utilize a very large, soft shadow: `0px 12px 48px rgba(14, 165, 233, 0.08)`. The slight blue tint in the shadow helps the surface feel integrated with the primary accent color.
- **Glassmorphism:** Overlays, such as the top navigation bar and dropdown menus, use a `16px` backdrop blur with an `85%` opacity white fill. This maintains a sense of place within the dashboard.
- **Borders:** Surfaces should use a subtle `1px` solid border in `#e2e8f0` to define edges when shadows overlap or in high-brightness environments.

## Shapes

The "Bento" feel is defined by extreme, friendly roundedness. 

- **Primary Containers:** All bento tiles and cards must use a **32px** (2rem) corner radius. This is the hallmark of the design system.
- **Interactive Elements:** Buttons, input fields, and tags utilize a **Pill** shape (fully rounded) to clearly distinguish them as interactive objects within the structural tiles.
- **Inner Elements:** Components nested inside cards (like image thumbnails or internal segments) should use a **12px** radius to maintain a nested geometric harmony.

## Components

### Bento Cards
The fundamental unit. Must include `32px` of internal padding. Headers within cards should use the `label-bold` style for categorization and `headline-md` for the title.

### Buttons
Primary buttons use a 135-degree linear gradient from `#0ea5e9` to `#38bdf8`. They are always pill-shaped. Secondary buttons use a transparent background with a navy text and a subtle 1px border.

### Input Fields
Inputs are pill-shaped with an `#f8fafc` background. Upon focus, they transition to a white background with a 1px primary sky-blue border and a soft blue outer glow.

### Navigation
The top navigation bar is `72px` tall, fixed to the top, and utilizes a backdrop blur. Navigation items indicate the "active" state with a soft blue background pill at 10% opacity.

### Status Chips
Status indicators are small, pill-shaped tags. Use the status colors with 10% background opacity and 100% foreground text opacity for a modern, high-fidelity look.