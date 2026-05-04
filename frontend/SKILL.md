# Law Firm Platform - Frontend UI/UX Design Skill

This skill defines comprehensive design guidelines for a first-time user interaction platform built for law firms. The design prioritizes professionalism, trust, clarity, security, and accessibility while maintaining a modern, approachable interface for lawyers, paralegals, administrative staff, and clients.

## Design Philosophy

**Core Principles:**

- **Trust & Credibility**: Professional, clean design that conveys reliability and competence
- **Clarity**: Clear information hierarchy, unambiguous actions, and transparent workflows
- **Security First**: Visual indicators of data protection, encrypted communications, audit trails
- **Inclusive Access**: Accessible to diverse users (aging lawyers, busy paralegals, tech-hesitant clients)
- **Efficiency**: Reduce clicks, streamline workflows, smart defaults for power users
- **Compliance Ready**: Design supports audit trails, record-keeping, and regulatory compliance
- **Calm Interface**: Reduced cognitive load through consistent patterns and logical organization

## Color Palette & Tokens

### Primary Colors

- **Professional Blue**: `#003366` (primary actions, key UI, trust/stability)
- **Light Blue**: `#e8f2ff` (backgrounds, information states)
- **Dark Blue**: `#001f3f` (text, headers, emphasis)
- **Accent Teal**: `#0088cc` (secondary actions, highlights, links)
- **Warm Neutral**: `#8b7355` (warm accents, premium feel)

### Status & Semantic Colors

- **Success Green**: `#2d7f3e` (approved, completed, valid)
- **Warning Orange**: `#d97706` (pending review, attention needed)
- **Error Red**: `#b91c1c` (errors, rejections, critical alerts)
- **Info Blue**: `#0369a1` (information, documentation, help)
- **Neutral Gray**: `#6b7280` (secondary text, disabled states)

### Neutral Palette

- **White**: `#ffffff` (primary background)
- **Off-White**: `#f9fafb` (secondary backgrounds, subtle contrast)
- **Light Gray**: `#e5e7eb` (borders, dividers)
- **Medium Gray**: `#9ca3af` (secondary text, muted UI)
- **Dark Gray**: `#1f2937` (primary text, dark mode ready)
- **Black**: `#111827` (maximum contrast headers)

## Typography

### Font Families

- **Serif Font** (`Georgia`, `"Times New Roman"`, serif): Headings, legal documents, case names
  - Conveys professionalism, tradition, and legal authority
  - Use for h1, h2, case titles, document headers
- **Sans-Serif Font** (`-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif`): Body, UI labels, forms
  - Modern, readable, accessible for screen readers and dyslexic users
  - Use for body text, navigation, buttons, form fields

### Font Scales

- **Page Title (h1)**: 32px serif, 700 weight, -0.5px letter-spacing, 1.3 line-height
- **Section Header (h2)**: 24px serif, 700 weight, 1.4 line-height
- **Subsection (h3)**: 18px serif, 600 weight, 1.5 line-height
- **Body Large**: 16px sans-serif, 400 weight, 1.6 line-height (readable for aging lawyers)
- **Body Regular**: 14px sans-serif, 400 weight, 1.6 line-height
- **Small Text**: 12px sans-serif, 400 weight, 1.5 line-height (secondary info)
- **X-Small**: 11px sans-serif, 500 weight (labels, badges, timestamps)
- **Monospace**: `Monaco`, `Courier New` for legal codes, case numbers, file references

## Spacing System

Use 8px base unit for professional breathing room:

- **xs**: 4px (tight spacing between inline elements)
- **sm**: 8px (internal padding, small gaps)
- **md**: 12px (medium gaps, form spacing)
- **lg**: 16px (section padding, primary gaps)
- **xl**: 24px (major section separation)
- **2xl**: 32px (page margins, significant breaks)
- **3xl**: 48px (hero sections, major divisions)

### Common Spacing Patterns

- **Card padding**: 16-20px
- **Section padding**: 24px vertical, 20px horizontal
- **Form field spacing**: 12px between fields
- **List item spacing**: 12px between items
- **Document margins**: 24px top/bottom, 20px left/right

## Spacing System

Use 4px base unit for consistent rhythm:

- **xs**: 4px
- **sm**: 8px
- **md**: 12px
- **lg**: 16px
- **xl**: 24px
- **2xl**: 32px
- **3xl**: 40px

### Common Spacing Values

- **Card padding**: 12-16px
- **Section padding**: 16px
- **Layout gap**: 12-24px
- **Component gap**: 6-12px

## Border Radius Scale

Use subtle, professional rounded corners:

- **none**: 0px (legal documents, formal tables)
- **sm**: 2px (minimal softness, form inputs)
- **md**: 4px (cards, standard components)
- **lg**: 6px (larger cards, modals)
- **full**: 50% (user avatars, circular status indicators only)

**Rationale**: Sharp corners convey formality and precision (important for legal work); avoid excessive rounding that feels casual.

## Shadow System

Understated shadows convey elevation and hierarchy without drawing attention:

- **shadow-none**: No shadow (documents, content cards)
- **shadow-xs**: `0 1px 2px rgba(0, 0, 0, 0.05)` (subtle hover effects)
- **shadow-sm**: `0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)` (cards, buttons)
- **shadow-md**: `0 4px 6px rgba(0, 0, 0, 0.1)` (floating panels, modals)
- **shadow-lg**: `0 10px 15px rgba(0, 0, 0, 0.1)` (critical modals, overlays)
- **Focus Ring**: `0 0 0 3px rgba(0, 51, 102, 0.1)` (blue focus) + shadow-sm
- **Security Indicator**: Subtle gold/bronze glow for encrypted/verified elements

## Component Patterns

### Primary Action Buttons

**Main CTA Button** (Save, Submit, File, Send)

```tsx
<button className="btn btn-primary" type="submit">
  <IconComponent />
  Save Document
</button>
```

- Background: Professional Blue (#003366)
- Text: White, sans-serif, 14px, 600 weight
- Padding: 12px 20px (generous touch target for older users)
- Border radius: 2px (professional, not rounded)
- Hover: Darker shade (#001a33), subtle shadow-xs
- Disabled: 50% opacity, cursor not-allowed
- Min width: 120px for accessibility

**Secondary Button** (Cancel, Back, Help)

```tsx
<button className="btn btn-secondary">Cancel</button>
```

- Background: Off-white (#f9fafb)
- Border: 1px solid Light Gray (#e5e7eb)
- Text: Dark Gray (#1f2937)
- Hover: Light Gray background, border Teal
- Padding: 12px 20px
- Font: 14px sans-serif

**Danger Button** (Delete, Archive, Reject)

```tsx
<button className="btn btn-danger">Delete Case</button>
```

- Background: Error Red (#b91c1c)
- Text: White
- Hover: Darker red (#991b1b)
- Requires confirmation modal

**Link Button** (Embedded actions)

```tsx
<button className="btn btn-link">View Details</button>
```

- Background: None
- Text: Teal (#0088cc), underlined on hover
- Font: 14px sans-serif, 500 weight
- No padding or border

### Cards & Document Containers

**Case Card** (Dashboard, case list)

```tsx
<div className="card card-case">
  <div className="card-header">
    <h3>Case Title / Case Number</h3>
    <span className="status-badge">Active</span>
  </div>
  <div className="card-body">
    <p className="case-info">Client Name · Matter Code</p>
    <p className="case-info-secondary">Next deadline: Jan 15, 2025</p>
  </div>
  <div className="card-footer">
    <span className="case-meta">Last edited: 2 hours ago</span>
  </div>
</div>
```

- Background: White
- Border: 1px solid Light Gray (#e5e7eb)
- Border radius: 4px (md)
- Padding: 16px
- Hover: shadow-xs, border Teal
- Title: 16px serif, 700 weight
- Secondary text: 12px, Medium Gray
- Full width responsive

**Document Card** (File browser)

```tsx
<div className="card card-document">
  <div className="document-icon">
    <FileIcon type="pdf" />
  </div>
  <div className="card-body">
    <h4>Document Name.pdf</h4>
    <p>Filed Jan 10, 2025 · 2.4 MB · By John Smith</p>
  </div>
  <div className="card-actions">
    <button className="btn-icon" title="Download">
      ⬇
    </button>
    <button className="btn-icon" title="Preview">
      👁
    </button>
  </div>
</div>
```

- Horizontal layout with icon, content, actions
- Icon: 32px, gray background
- Body: Flex 1
- Actions: Compact icon buttons (32px × 32px)
- Hover: Shadow-xs, light background

**Client Profile Card**

```tsx
<div className="card card-profile">
  <img src="avatar" alt="Client" className="profile-photo" />
  <h4>Client Name</h4>
  <p className="role">Contact Person</p>
  <p className="info">john@lawfirm.com · (555) 123-4567</p>
  <div className="profile-actions">
    <button className="btn btn-secondary">Email</button>
    <button className="btn btn-secondary">Call</button>
  </div>
</div>
```

- Centered layout
- Profile photo: 60px circle
- Background: Off-white
- Max width: 300px
- Good for sidebar or modal

### Status & Security Badges

**Case Status Badge**

```tsx
<span className={`status-badge status-${status}`}>{statusLabel}</span>
```

- Classes: `status-active`, `status-pending`, `status-completed`, `status-archived`, `status-on-hold`
- Padding: 6px 12px
- Font: 11px sans-serif, 600 weight
- Border radius: 2px
- Color coding:
  - Active: Teal background, dark text
  - Pending: Orange background, dark text
  - Completed: Green background, white text
  - Archived: Gray background, dark text

**Security/Encryption Badge**

```tsx
<span className="badge-secure">
  <LockIcon /> Encrypted
</span>
```

- Background: Light teal
- Text: Teal (#0088cc)
- Icon: 12px
- Font: 11px, 600 weight
- Used for secure messaging, encrypted documents

**Verification Badge**

```tsx
<span className="badge-verified">
  <CheckIcon /> Verified
</span>
```

- Background: Light green
- Text: Success Green
- Icon: 12px
- Font: 11px, 600 weight

### Forms & Input Fields

**Text Input / Textarea**

```tsx
<div className="form-group">
  <label htmlFor="client-name">Client Name *</label>
  <input
    type="text"
    id="client-name"
    placeholder="Enter full legal name"
    required
  />
  <p className="form-hint">Must match identification documents</p>
</div>
```

- Label: 13px sans-serif, 500 weight, Dark Gray
- Input: 14px sans-serif, padding 10px 12px
- Border: 1px solid Light Gray (#e5e7eb)
- Border radius: 2px
- Focus: Blue border (#003366), focus ring
- Placeholder: Medium Gray, italicized
- Hint text: 12px, secondary color
- Min height: 44px for touch accessibility
- Required indicator: Red asterisk

**Form Validation**

```tsx
<input className="input-error" />
<p className="error-message">This field is required</p>

<input className="input-success" />
<p className="success-message">Email verified</p>
```

- Error: Red border (#b91c1c), red text error message
- Success: Green border (#2d7f3e), green checkmark
- Warning: Orange border, orange text
- Message font: 12px, appropriate color

**Select Dropdown**

```tsx
<select className="form-select">
  <option>Select a matter type...</option>
  <option>Corporate</option>
  <option>Litigation</option>
</select>
```

- Similar styling to text input
- Padding: 10px 12px with dropdown icon (16px right padding)
- Custom styling to match form aesthetic
- Focus state: Blue border + ring

**Checkbox & Radio**

```tsx
<div className="form-check">
  <input type="checkbox" id="agree" />
  <label htmlFor="agree">I agree to the terms of service</label>
</div>
```

- Checkbox/Radio: 18px × 18px (touch-friendly)
- Border: 1px solid Light Gray
- Checked: Professional Blue background
- Label: 14px, adjacent to control
- Spacing: 8px between control and label

### Navigation & Header

**Top Navigation Bar**

```tsx
<nav className="topnav">
  <div className="nav-logo">
    <Logo /> <!-- Law firm logo -->
  </div>
  <div className="nav-primary">
    <a href="/dashboard">Dashboard</a>
    <a href="/cases">Cases</a>
    <a href="/clients">Clients</a>
    <a href="/documents">Documents</a>
  </div>
  <div className="nav-secondary">
    <button className="btn-notification">🔔</button>
    <button className="btn-user-menu">👤</button>
  </div>
</nav>
```

- Height: 64px fixed
- Background: White with subtle bottom border
- Logo: 32px square, left aligned with 20px padding
- Primary nav: Horizontal links, Dark Gray text, Blue underline on active/hover
- Font: 14px sans-serif, 500 weight
- Link spacing: 24px horizontal gap
- Secondary actions: Right aligned
- Sticky positioning at top

**Sidebar Navigation** (Left panel)

```tsx
<aside className="sidebar">
  <div className="sidebar-section">
    <h3>My Cases</h3>
    <ul className="nav-list">
      <li>
        <a href="#" className="nav-link active">
          Case Name
        </a>
      </li>
      <li>
        <a href="#" className="nav-link">
          Another Case
        </a>
      </li>
    </ul>
  </div>
</aside>
```

- Width: 280px fixed
- Background: Off-white (#f9fafb) or white
- Border right: 1px Light Gray
- Section padding: 16px
- Section header: 12px sans-serif, 600 weight, uppercase, Medium Gray
- Link: 14px sans-serif, 400 weight
- Active link: Professional Blue text, left border (3px)
- Hover: Light Blue background
- Scrollable if content exceeds height

**Breadcrumb Navigation**

```tsx
<nav className="breadcrumbs">
  <a href="/cases">Cases</a>
  <span className="separator">/</span>
  <a href="/cases/123">Smith v. Jones</a>
  <span className="separator">/</span>
  <span className="current">Documents</span>
</nav>
```

- Font: 12px sans-serif
- Text: Medium Gray
- Current: Dark Gray, not a link
- Separator: `/` or `>`
- Padding: 8px 0

### Document & Data Tables

**Case Activity Table**

```tsx
<table className="table">
  <thead>
    <tr>
      <th>Date</th>
      <th>Activity</th>
      <th>By</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Jan 10, 2025</td>
      <td>Motion Filed</td>
      <td>John Smith</td>
      <td>
        <span className="status-badge">Filed</span>
      </td>
    </tr>
  </tbody>
</table>
```

- Header: Professional Blue background, white text, 13px 600 weight
- Rows: White background, bordered or zebra striped (alternate light gray)
- Row height: 44px minimum
- Padding: 12px per cell
- Borders: Subtle Light Gray
- Hover row: Light Blue background
- Responsive: Horizontal scroll on mobile

**Case Timeline**

```tsx
<div className="timeline">
  <div className="timeline-item">
    <div className="timeline-marker"></div>
    <div className="timeline-content">
      <p className="timeline-date">Jan 15, 2025</p>
      <p className="timeline-title">Court Date Scheduled</p>
      <p className="timeline-description">Assigned to Judge Martinez</p>
    </div>
  </div>
</div>
```

- Vertical line (1px Light Gray)
- Marker: 12px circle, color indicates status (Blue pending, Green completed)
- Content: Left aligned, 16px gap from marker
- Date: 11px sans-serif, Medium Gray
- Title: 14px sans-serif, 600 weight
- Description: 12px sans-serif, secondary gray

### Modals & Dialogs

**Confirmation Modal**

```tsx
<div className="modal">
  <div className="modal-content">
    <div className="modal-header">
      <h2>Delete Case?</h2>
      <button className="btn-close" aria-label="Close"></button>
    </div>
    <div className="modal-body">
      <p>
        This action cannot be undone. All case data will be permanently deleted.
      </p>
    </div>
    <div className="modal-footer">
      <button className="btn btn-secondary">Cancel</button>
      <button className="btn btn-danger">Delete Case</button>
    </div>
  </div>
</div>
```

- Background overlay: Rgba(0, 0, 0, 0.5) semi-transparent
- Modal width: 500px max, 90% on mobile
- Header: Dark Gray, 18px serif, 700 weight
- Close button: X icon, top right, 32px × 32px
- Body: 14px sans-serif, padding 20px
- Footer: Right aligned buttons, 12px gap
- Shadow: shadow-lg
- Padding: 24px
- Border radius: 4px

**Document Preview Modal**

- Full-screen or 80% viewport
- Close button prominent
- Document centered in scrollable container
- Toolbar: Download, print, share buttons at top
- Metadata: Document name, date, size at bottom

### Search & Filters

**Search Bar**

```tsx
<div className="search-container">
  <input
    type="search"
    placeholder="Search cases, clients, documents..."
    className="search-input"
  />
  <button className="btn-search" aria-label="Search">
    🔍
  </button>
</div>
```

- Background: White
- Border: 1px Light Gray
- Padding: 10px 12px
- Font: 14px sans-serif
- Width: Full or max 400px
- Dropdown results: Grouped by type (cases, clients, documents), max 8 results

**Filter Panel**

```tsx
<div className="filters">
  <div className="filter-group">
    <h4>Status</h4>
    <label>
      <input type="checkbox" /> Active
    </label>
    <label>
      <input type="checkbox" /> Pending
    </label>
  </div>
  <div className="filter-group">
    <h4>Date Range</h4>
    <input type="date" />
  </div>
  <button className="btn btn-primary">Apply Filters</button>
</div>
```

- Sidebar or collapsible panel
- Group headings: 13px sans-serif, 600 weight
- Options: Checkboxes or radio buttons
- Spacing: 12px between groups
- Apply/Reset buttons: Full width

### Message Bubbles

**User Message**

- Background: Blue (#1a73e8)
- Text: White
- Border radius: 12px with asymmetric corners
- Padding: 12px 16px
- Max width: ~70% of container
- Alignment: Right aligned
- Font size: 15px

**Assistant Message**

- Layout: flex, gap 12px
- Avatar: 32px circle with logo
- Body: flex 1
- Contains: Metadata chips, answer text, expandable sections
- Max width: ~70% of container

**Typing Indicator**

- Animated dots (3 dots with staggered animation)
- Dot size: 8px
- Animation: scale 0.8 → 1, opacity pulse
- Gap between dots: 4px

### Form Elements & Error States

**Error Banner**

```tsx
<div className="error-banner">
  <ErrorIcon />
  {error message}
</div>
```

- Background: Light red/orange
- Border: 1px solid error color
- Padding: 12px 16px
- Border radius: 8px
- Display: flex, gap 12px, align center
- Icon: 16px

## Accessibility Guidelines

### WCAG AA Compliance (Mandatory)

- All text must meet WCAG AA standards (4.5:1 contrast for normal text, 3:1 for large text)
- Links must be underlined or have distinct visual indication (not color alone)
- Focus indicators must be clearly visible (3px blue ring minimum)
- Color should never be the only way to convey information

### Keyboard Navigation (Critical for Lawyers)

- All interactive elements must be fully keyboard accessible
- Tab order follows logical visual flow (left-to-right, top-to-bottom)
- Escape key closes modals and dismisses overlays
- Enter/Space activates buttons
- Arrow keys navigate lists and menus
- No keyboard traps
- Focus visible at all times with minimum 2px indicator

### ARIA Labels & Semantic HTML

```tsx
// Always use semantic HTML
<nav> for navigation
<main> for main content
<aside> for sidebars
<section> for major content blocks
<article> for case documents, posts
<form> for forms
<button> for clickable actions
<a> for navigation links

// Add ARIA where HTML semantics insufficient
<button aria-label="Delete case" aria-describedby="delete-warning">
  🗑
</button>

// Announce loading states
<div aria-live="polite" aria-busy="true">Loading case details...</div>

// Mark required fields
<label>
  Client Name <span aria-label="required">*</span>
</label>
```

### Images & Alt Text

```tsx
// Descriptive alt text for meaningful images
<img alt="John Smith, Lead Attorney" src="john-photo.jpg" />

// Decorative images: empty alt
<img alt="" aria-hidden="true" src="decorative-line.svg" />

// Icons with text need alt, buttons need aria-label
<button aria-label="Download case documents">
  <DownloadIcon />
</button>

// PDFs and document previews
<img alt="Motion for Summary Judgment - Smith v. Jones, filed Jan 10 2025" src="motion.pdf" />
```

### Focus Management

- Focus outline: 3px solid Professional Blue (#003366) with 2px offset
- Focus visible in all browsers (use `:focus-visible` with fallback to `:focus`)
- When opening modals, move focus to first interactive element
- When closing modals, return focus to trigger button
- Skip links for keyboard users: "Skip to main content"

### Motion & Animation (Important for Accessibility)

```css
/* Respect prefers-reduced-motion */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

- No flashing or strobing (>2Hz)
- Animations should be < 300ms
- Provide pause/stop controls for auto-playing content
- Clear animations that convey information

### Text Readability

- Line length: 60-80 characters for optimal reading
- Line height: 1.5-1.6 minimum
- Font size: 14px minimum for body text (16px preferred for older users)
- Letter spacing: -0.2px to 0.5px, never negative
- Sufficient color contrast: 4.5:1 for normal text

### Aging User Considerations (Lawyers are older on average)

- Larger click/touch targets: minimum 44px × 44px
- Readable font sizes: 14px+ for body text
- High contrast design (avoid gray-on-gray)
- Clear labels and instructions
- Simple, consistent navigation
- Avoid time limits unless necessary (extend to 5+ minutes)
- Error prevention: Confirmation before destructive actions
- Help & documentation easily accessible

### Form Accessibility

```tsx
// Always associate labels with inputs
<label htmlFor="case-number">Case Number</label>
<input id="case-number" type="text" aria-required="true" />

// Group related fields
<fieldset>
  <legend>Case Details</legend>
  {/* form fields */}
</fieldset>

// Required fields clearly marked
<span aria-label="required" className="required-indicator">*</span>

// Error messages linked to inputs
<input aria-describedby="email-error" />
<p id="email-error" className="error-message">Invalid email format</p>
```

### Screen Reader Testing

- Test with NVDA (Windows), JAWS (Windows), or VoiceOver (Mac)
- Verify all content is announced
- Check form labels are properly associated
- Confirm headings have proper hierarchy (no skipped levels)
- Test with assistive technology users in mind

## Layout Patterns & Information Architecture

### Main Dashboard Layout

```
┌─────────────────────────────────────────────┐
│  Top Nav (64px) - Logo, Primary Nav, Account │
├─────────────────────────────────────────────┤
│ Sidebar (280px) │           Main Content     │
│ My Cases        │  Dashboard/Case Details   │
│ Clients         │                           │
│ Documents       │                           │
│ Reports         │                           │
└─────────────────────────────────────────────┘
```

**CSS Grid Structure:**

```css
.app-container {
  display: grid;
  grid-template-columns: 280px 1fr;
  grid-template-rows: 64px 1fr;
  min-height: 100vh;
}

.topnav {
  grid-column: 1 / -1;
  grid-row: 1;
}

.sidebar {
  grid-column: 1;
  grid-row: 2;
  overflow-y: auto;
}

.main-content {
  grid-column: 2;
  grid-row: 2;
  overflow-y: auto;
  padding: 24px;
}
```

### Case Detail Page Layout

```
┌────────────────────────────────────────────┐
│ Case Name / Case Number                     │
│ [Active] Client: ABC Corp | Matter: 2024-001│
├────────────────────────────────────────────┤
│ Tabs: Overview | Documents | Activity       │
├────────────────────────────────────────────┤
│  Overview Tab:                              │
│  ├─ Key Information Grid                    │
│  ├─ Next Actions Card                       │
│  ├─ Team Members                            │
│  └─ Recent Documents                        │
│                                              │
│  Documents Tab:                             │
│  ├─ Filter & Search                         │
│  └─ Document List (Table)                   │
│                                              │
│  Activity Tab:                              │
│  └─ Timeline of Case Events                 │
└────────────────────────────────────────────┘
```

### Two-Column Form Layout

```
┌─────────────────────────────────────┐
│ Form Title                          │
├─────────────────────────────────────┤
│ Col 1              │   Col 2        │
│ ┌───────────┐      │ ┌───────────┐ │
│ │ Field 1   │      │ │ Field 2   │ │
│ └───────────┘      │ └───────────┘ │
│                    │                │
│ ┌────────────────┐ │                │
│ │  Full Width    │ │                │
│ │  Field 3       │ │                │
│ └────────────────┘ │                │
└─────────────────────────────────────┘
```

### Sidebar Navigation Structure

```
My Cases
  ├─ Active (5)
  │  ├─ Smith v. Jones
  │  ├─ Estate of Johnson
  │  └─ ...
  ├─ Pending (2)
  │  ├─ Initial Consultation
  │  └─ Contract Review

Clients
  ├─ Individuals (23)
  ├─ Corporations (8)
  └─ Search All

Documents
  ├─ Recent
  ├─ By Case
  └─ By Type

Reports
  └─ Billing
```

### Mobile Layout (< 768px)

- Stack layout: Sidebar collapses to hamburger menu
- Top nav: Logo, hamburger menu, user account only
- Main content: Full width
- Forms: Single column
- Tables: Horizontal scroll or card view
- Modals: Full width with padding

## Responsive Design Strategy

**Breakpoints:**

- Mobile: < 640px (phones)
- Tablet: 640px - 1024px (iPads, small laptops)
- Desktop: > 1024px (standard monitors)
- Large: > 1440px (wide screens)

**Mobile-First Adjustments:**

- Single column layout
- Full-width buttons and inputs
- Touch-friendly sizing (44px minimum)
- Hamburger menu for navigation
- Bottom sheet modals instead of centered
- Card view for tables/lists

## Branding & Professional Identity

### Law Firm Logo & Branding

- **Logo placement**: Top-left of navigation (32-40px height in header)
- **Logo usage**: Monochrome or brand color, never distorted or rotated
- **Logo clear space**: Minimum 12px margin on all sides
- **Wordmark**: Firm name in serif font, professional shade
- **Tagline/Motto**: Optional, if present use small sans-serif below logo

### Professional Typography Strategy

- **Serif (Georgia, Times New Roman)** for:
  - Page headings (convey authority and formality)
  - Case names and titles
  - Document headers
  - Important labels
- **Sans-serif (System stack)** for:
  - Body text (readability)
  - Form labels
  - Navigation
  - Data displays
  - UI controls

### Color Application

- **Professional Blue**: Primary actions, active states, focus indicators
- **Dark Blue**: Headers, primary text, emphasis
- **Teal**: Links, secondary actions, hover states
- **Neutral grays**: Supporting text, disabled states, borders
- **Status colors**: Green (approved/complete), Orange (pending), Red (error/urgent)

### Visual Hierarchy

1. **Headings**: Serif, large, dark color (case names, page titles)
2. **Subheadings**: Serif, medium, dark blue
3. **Body text**: Sans-serif, 14px, dark gray
4. **Secondary text**: Sans-serif, 12px, medium gray
5. **Labels & metadata**: Sans-serif, 11px, medium gray
6. **Disabled/muted**: Sans-serif, 11px, light gray, 50% opacity

### Trust & Credibility Design Elements

- **Consistent spacing**: Professional rhythm, not cramped
- **Subtle shadows**: Convey depth without distraction
- **Clear CTAs**: Important actions are obvious but not aggressive
- **Security indicators**: Lock icons, "Encrypted" badges for sensitive data
- **Last updated timestamps**: Show data freshness and reliability
- **User confirmation modals**: For any destructive actions
- **Audit trail visibility**: Show when/who/what changed (where applicable)

### Tone & Voice

- **Professional**: Formal language, legal accuracy
- **Clear**: Simple explanations of complex legal concepts
- **Respectful**: Acknowledge lawyers' expertise, don't patronize
- **Helpful**: Provide guidance without being intrusive
- **Transparent**: Clear about limitations, confirmations, and consequences
- **Action-oriented**: CTA copy should be specific ("Save Document" not "OK")

## Law Firm Specific Patterns & Best Practices

### Case Management Dashboard

**Empty State:**

```tsx
<div className="empty-state">
  <div className="empty-icon">📋</div>
  <h3>No Cases Yet</h3>
  <p>Create a new case or ask your administrator for access</p>
  <button className="btn btn-primary">+ New Case</button>
</div>
```

- Icon + title + description + action
- Centered layout
- Encourage next action

### Document Management Patterns

**Upload Area:**

```tsx
<div className="upload-zone">
  <svg className="upload-icon">⬆</svg>
  <h4>Drop documents here</h4>
  <p>
    or <button>browse files</button>
  </p>
  <p className="upload-hint">Accepted: PDF, DOCX, TXT (max 50MB)</p>
</div>
```

- Drag-and-drop enabled
- Clear file type guidance
- Size limits stated
- Fallback browse option

**Document Metadata Display:**

```tsx
<div className="document-meta">
  <p>
    <strong>Uploaded:</strong> Jan 10, 2025 by John Smith
  </p>
  <p>
    <strong>File Size:</strong> 2.4 MB
  </p>
  <p>
    <strong>Pages:</strong> 18
  </p>
  <p>
    <strong>Last Reviewed:</strong> Jan 15, 2025
  </p>
</div>
```

- Label + value pairs
- Timestamp for audit trail
- File properties clearly visible

### Time-Sensitive Information

**Deadline/Deadline Alert Card:**

```tsx
<div className="alert alert-deadline">
  <svg className="alert-icon">⚠️</svg>
  <div>
    <h4>Upcoming Deadline</h4>
    <p>
      Motion due: <strong>January 31, 2025</strong> (15 days)
    </p>
    <p className="alert-description">Court order requires response</p>
  </div>
</div>
```

- Color-coded by urgency (red for <7 days, orange for <14 days)
- Clear deadline date
- Context about why it matters
- Action button if applicable

### Authentication & Security

**Login Form:**

```tsx
<form className="login-form">
  <div className="form-logo">
    <!-- Firm Logo -->
  </div>
  <h1>Law Firm Portal</h1>
  <div className="form-group">
    <label>Email Address</label>
    <input type="email" required />
  </div>
  <div className="form-group">
    <label>Password</label>
    <input type="password" required />
  </div>
  <button className="btn btn-primary" type="submit">Sign In</button>
  <a href="/forgot-password">Forgot your password?</a>
</form>
```

- Firm branding at top
- Email + password fields
- Password recovery link
- Optional: 2FA notification

**Multi-Factor Authentication:**

```tsx
<div className="mfa-container">
  <h2>Verify Your Identity</h2>
  <p>Enter the 6-digit code from your authenticator app</p>
  <input type="text" maxLength="6" placeholder="000000" />
  <button className="btn btn-primary">Verify</button>
  <p className="mfa-help">
    <a href="#">Didn't receive a code?</a>
  </p>
</div>
```

- Clear instructions
- Single input field (6 digits)
- Backup option link
- High contrast for security

### Data Export Patterns

**Export Options Modal:**

```tsx
<div className="modal">
  <h2>Export Case Data</h2>
  <div className="export-options">
    <label>
      <input type="radio" /> PDF Report
    </label>
    <label>
      <input type="radio" /> Excel (Documents)
    </label>
    <label>
      <input type="radio" /> CSV (Activity Log)
    </label>
    <label>
      <input type="checkbox" /> Include confidential notes
    </label>
  </div>
  <p className="form-hint">Data will be encrypted and available for 7 days</p>
  <div className="modal-footer">
    <button className="btn btn-secondary">Cancel</button>
    <button className="btn btn-primary">Generate Export</button>
  </div>
</div>
```

- Format options (PDF most common for lawyers)
- Include/exclude options
- Confirmation of what will be exported
- Timeline for availability

### Notification & Alert System

**Toast Notifications (Non-intrusive):**

```tsx
<div className="toast toast-success">
  <svg className="toast-icon">✓</svg>
  <p>Document saved successfully</p>
  <button className="btn-close" aria-label="Dismiss"></button>
</div>
```

- Bottom right position (doesn't block critical content)
- Auto-dismiss after 5 seconds
- Manual close option
- Color-coded: green (success), red (error), blue (info), orange (warning)

**In-Line Alerts (For forms):**

```tsx
<div className="alert alert-warning">
  <p>
    <strong>Warning:</strong> This case will be archived in 30 days if no
    activity occurs.
  </p>
</div>
```

- Alert box with border and background
- Clear warning/error/info level
- Specific, actionable message
- Link to action if available ("Learn more", "Dismiss this warning")

### Pagination & Large Datasets

**Table Pagination:**

```tsx
<div className="pagination">
  <button className="btn-pagination" disabled>
    &lt; Previous
  </button>
  <span className="pagination-info">Showing 1-25 of 487</span>
  <select className="pagination-select">
    <option>25 per page</option>
    <option>50 per page</option>
    <option>100 per page</option>
  </select>
  <button className="btn-pagination">Next &gt;</button>
</div>
```

- Previous/Next buttons
- Row count display
- Items per page selector
- Jump to page input (for large datasets)

### Print-Friendly Layouts

**Print Stylesheet Considerations:**

```css
@media print {
  .no-print {
    display: none;
  }
  body {
    background: white;
    color: black;
  }
  .sidebar {
    display: none;
  }
  .main-content {
    margin: 0;
  }
  a {
    color: inherit;
    text-decoration: underline;
  }
  .page-break {
    page-break-after: always;
  }
}
```

- Hide unnecessary UI (navigation, sidebars)
- Preserve document structure
- Dark text on white for printing
- Page breaks for long documents
- Show full URLs for links

## Code Standards & Implementation Guidelines

### CSS Architecture

```css
/* 1. Reset & Variables */
:root {
  --color-primary-blue: #003366;
  --color-dark-blue: #001f3f;
  --color-teal: #0088cc;
  --color-success: #2d7f3e;
  --spacing-base: 8px;
}

/* 2. Base Styles */
html, body {
  font-family: system sans-serif;
  color: var(--color-dark-blue);
}

/* 3. Layout Components */
.topnav { ... }
.sidebar { ... }
.main-content { ... }

/* 4. Feature Components */
.case-card { ... }
.button { ... }

/* 5. Utility Classes */
.text-muted { ... }
.m-0 { margin: 0; }
```

### BEM Naming Convention

- `.block`: Component name (`.case-card`, `.upload-zone`)
- `.block__element`: Child element (`.case-card__header`, `.case-card__actions`)
- `.block--modifier`: State or variation (`.btn--primary`, `.status--active`)

**Examples:**

```css
.case-card {
}
.case-card__title {
}
.case-card__footer {
}
.case-card--highlighted {
}
.status-badge {
}
.status-badge--active {
}
```

### Responsive Design Implementation

```css
/* Mobile-first approach */
.main-layout {
  display: flex;
  flex-direction: column;
}

/* Tablet and up */
@media (min-width: 768px) {
  .main-layout {
    flex-direction: row;
  }
  .sidebar {
    width: 280px;
  }
}

/* Large screens */
@media (min-width: 1440px) {
  .container {
    max-width: 1200px;
    margin: 0 auto;
  }
}
```

### TypeScript/React Best Practices

```tsx
// 1. Type definitions for law firm data
interface Case {
  id: string;
  caseNumber: string;
  caseTitle: string;
  clientName: string;
  status: "active" | "pending" | "completed" | "archived";
  lastModified: Date;
}

interface User {
  id: string;
  name: string;
  email: string;
  role: "attorney" | "paralegal" | "admin" | "client";
}

// 2. Accessible button component
interface ButtonProps {
  onClick?: () => void;
  children: React.ReactNode;
  variant: "primary" | "secondary" | "danger";
  disabled?: boolean;
  ariaLabel?: string;
}

export const Button = ({
  onClick,
  children,
  variant,
  disabled,
  ariaLabel,
}: ButtonProps) => (
  <button
    className={`btn btn--${variant}`}
    onClick={onClick}
    disabled={disabled}
    aria-label={ariaLabel}
    aria-disabled={disabled}
  >
    {children}
  </button>
);

// 3. Form component with validation
const CaseForm = ({ onSubmit }: { onSubmit: (data: Case) => void }) => {
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    // Validation logic
    // Submit with error handling
  };

  return (
    <form onSubmit={handleSubmit} noValidate>
      {/* form fields with error messages */}
    </form>
  );
};
```

### Accessibility in Code

```tsx
// 1. Semantic HTML
<nav aria-label="Main navigation">
  <a href="/" aria-current="page">Dashboard</a>
</nav>

// 2. ARIA live regions for updates
<div aria-live="polite" aria-atomic="true">
  {status === 'saving' && 'Saving document...'}
  {status === 'saved' && 'Document saved successfully'}
</div>

// 3. Proper form structure
<form>
  <fieldset>
    <legend>Case Information</legend>
    <label htmlFor="case-number">
      Case Number <span className="required">*</span>
    </label>
    <input
      id="case-number"
      type="text"
      required
      aria-required="true"
      aria-describedby="case-hint"
    />
    <span id="case-hint" className="form-hint">
      Format: YYYY-000000
    </span>
  </fieldset>
</form>
```

### Performance Considerations

- Lazy load large document previews
- Implement virtual scrolling for long lists
- Cache frequently accessed case data
- Minimize bundle size (law firms may have older internet)
- Progressive enhancement (works without JavaScript for core functionality)
- Optimize images (PDFs, document screenshots)

### Testing Standards

```tsx
// Unit test for button component
test("Button calls onClick when clicked", () => {
  const mockClick = jest.fn();
  render(<Button onClick={mockClick}>Save</Button>);
  fireEvent.click(screen.getByText("Save"));
  expect(mockClick).toHaveBeenCalled();
});

// Accessibility test
test("Button has proper aria-label", () => {
  render(<Button ariaLabel="Save case">Save</Button>);
  expect(screen.getByRole("button")).toHaveAttribute("aria-label", "Save case");
});

// Form validation test
test("Form shows error for invalid input", () => {
  render(<CaseForm onSubmit={jest.fn()} />);
  fireEvent.submit(screen.getByRole("form"));
  expect(screen.getByText(/case number required/i)).toBeInTheDocument();
});
```

## Maintenance, Compliance & Quality Assurance

### Design System Maintenance

When updating this design system:

1. **Color changes**: Update CSS variables and test contrast ratios (WCAG AA minimum)
2. **Typography changes**: Test readability on older monitors at 96dpi
3. **Component changes**: Update all instances and test across browsers
4. **Breaking changes**: Document and communicate to team
5. **Versioning**: Maintain changelog of all updates

### Legal Compliance Checklist

- [ ] Data encryption indicators visible where applicable
- [ ] Audit trail timestamps on all sensitive changes
- [ ] Confirmation modals for destructive actions
- [ ] GDPR compliance (consent for tracking, right to delete)
- [ ] Data retention notices visible
- [ ] No personal client data in error messages (shown in logs, not UI)
- [ ] Secure password requirements enforced
- [ ] Session timeout warnings displayed
- [ ] Export/download audit trails available

### Accessibility Compliance Testing

- [ ] WCAG AA color contrast verified (4.5:1 normal, 3:1 large)
- [ ] Keyboard navigation fully functional (Tab, Shift+Tab, Enter, Escape)
- [ ] Screen reader testing with NVDA or JAWS (all content announced)
- [ ] Focus indicators clearly visible throughout
- [ ] No keyboard traps
- [ ] Form labels properly associated
- [ ] Images have descriptive alt text
- [ ] Videos have captions (if present)
- [ ] Page structure uses proper semantic HTML

### Browser & Device Testing

**Desktop Browsers (Minimum):**

- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Edge (latest version)

**Mobile/Tablet:**

- iOS Safari (iPad, iPhone)
- Android Chrome
- Android Firefox

**Testing Checklist:**

- [ ] Responsive layouts at all breakpoints
- [ ] Touch targets minimum 44px × 44px
- [ ] Form inputs zoom correctly (no auto-zoom on focus iOS issue)
- [ ] No horizontal scroll on mobile
- [ ] Modals work on small screens
- [ ] Navigation accessible on touch

### Performance Guidelines

- Page load: < 3 seconds on 3G connection
- First contentful paint: < 1.5 seconds
- Time to interactive: < 3.5 seconds
- Bundle size: < 500KB (gzipped) for core app
- Document preview: < 2 second load (lazy load PDFs)
- Case list load: < 1 second

**Tools for Testing:**

- Lighthouse (Chrome DevTools)
- WebPageTest
- GTmetrix
- Accessibility Checker (Axe DevTools)

### Security & Privacy

- [ ] No confidential client data in console logs
- [ ] No personal information in URLs (case IDs okay, case titles/names not)
- [ ] HTTPS enforced everywhere
- [ ] Secure password reset flow
- [ ] Session expiration after inactivity
- [ ] Error messages generic (don't reveal database structure)
- [ ] File uploads validated (type, size, content)
- [ ] Rate limiting on login/sensitive endpoints
- [ ] Regular security audits scheduled

### Version Control & Documentation

- Clear commit messages describing changes
- Document design rationale in comments
- Update this skill file with all pattern changes
- Maintain component library documentation
- Keep accessibility guidelines up-to-date
- Document any deviations from standards with justification

### User Testing & Feedback

- Conduct usability testing with actual lawyers/paralegals
- Test with older users (40+) for accessibility
- Gather feedback on workflows and efficiency
- A/B test major UI changes before rollout
- Monitor user support tickets for design issues
- Quarterly design review with stakeholder feedback
