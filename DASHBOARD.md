# LEO Optima Enterprise Dashboard

A professional, production-ready web dashboard for monitoring and managing LEO Optima's multi-tenant infrastructure. Built with React 19, Tailwind CSS 4, and inspired by industry leaders like Stripe and Vercel.

---

## Overview

The LEO Optima Dashboard provides a comprehensive interface for:
- **Real-time Metrics**: Monitor cost savings, token optimization, and cache performance
- **Multi-Tenant Management**: View and manage multiple tenant accounts with isolated analytics
- **API Key Management**: Securely generate and manage API keys for different service tiers
- **Performance Analytics**: Visualize optimization routes (Cache, Fast, Consensus) and trends
- **Usage Quotas**: Track token usage and cost limits per tenant

---

## Architecture

### Directory Structure

```
dashboard/
├── client/                 # React frontend application
│   ├── public/            # Static assets
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   │   ├── DashboardLayout.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── MetricCard.tsx
│   │   │   └── ui/        # shadcn/ui components
│   │   ├── pages/         # Page-level components
│   │   │   ├── Dashboard.tsx
│   │   │   └── NotFound.tsx
│   │   ├── contexts/      # React contexts
│   │   ├── hooks/         # Custom React hooks
│   │   ├── lib/           # Utility functions
│   │   ├── App.tsx        # Main router
│   │   ├── main.tsx       # React entry point
│   │   └── index.css      # Global styles and design tokens
│   └── index.html         # HTML template
├── server/                # Express server for production
│   └── index.ts           # Server entry point
├── package.json           # Dependencies and scripts
└── README.md              # This file
```

### Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend | React | 19.2.1 |
| Styling | Tailwind CSS | 4.1.14 |
| UI Components | shadcn/ui | Latest |
| Icons | Lucide React | 0.453.0 |
| Charts | Recharts | 2.15.2 |
| Routing | Wouter | 3.3.5 |
| Build Tool | Vite | 7.1.7 |
| Runtime | Node.js | 22.13.0 |

---

## Design System

### Color Palette

The dashboard uses a **Minimalist Precision** design philosophy inspired by Stripe and Vercel.

| Color | Hex | Usage |
|-------|-----|-------|
| Primary | `#3B82F6` | Buttons, links, accents |
| Navy | `#0F172A` | Sidebar background, text |
| Success | `#10B981` | Positive metrics, savings |
| Warning | `#F59E0B` | Quota warnings, alerts |
| Error | `#EF4444` | Critical issues |
| Neutral | `#E2E8F0` | Borders, backgrounds |

### Typography

- **Display/Headings**: Inter (700 weight) - Bold, modern, trustworthy
- **Body Text**: Inter (400-600 weight) - Clean, readable, professional
- **Code/Technical**: Fira Code (500 weight) - Monospace for API keys and data

### Spacing & Radius

- Base radius: `0.5rem` (8px)
- Spacing scale: 4px, 8px, 12px, 16px, 24px, 32px, 48px, 64px
- Shadows: Subtle (0.5px blur) for depth without distraction

---

## Features

### 1. Dashboard Page

The main landing page displays:

**Key Metrics (4-column grid)**
- **Total Cost Saved**: Cumulative savings with trend indicator
- **Tokens Optimized**: Total tokens reduced with percentage change
- **Cache Hit Rate**: Percentage of requests served from cache
- **Requests Processed**: Total requests handled today

**Visualizations**
- **Cost & Token Savings Chart**: Line chart showing cumulative savings over time
- **Request Routes Pie Chart**: Distribution of Cache, Fast, and Consensus routes
- **Multi-Tenant Performance Table**: Per-tenant metrics including requests, savings, and cache hit rates

### 2. Sidebar Navigation

Persistent left sidebar with:
- LEO Optima logo and branding
- Navigation links (Dashboard, Analytics, Metrics, Tenants, API Keys, Settings)
- Active state indicators
- Responsive collapse on mobile

### 3. Top Bar

Header with:
- Menu toggle for mobile
- Current tenant name and tier display
- Settings and logout buttons
- Real-time status indicators

---

## API Integration

The dashboard connects to the LEO Optima backend via the following endpoints:

### Authentication

All requests require the `X-API-Key` header:

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/v1/analytics
```

### Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/analytics` | GET | Fetch tenant analytics and metrics |
| `/v1/optimization/status` | GET | Get current optimization configuration |
| `/v1/optimization/cache/stats` | GET | Get cache statistics |
| `/v1/admin/tenants` | GET | List all tenants (admin only) |
| `/v1/admin/tenants` | POST | Create new tenant (admin only) |

### Example: Fetching Analytics

```typescript
const response = await fetch('http://localhost:8000/v1/analytics', {
  headers: {
    'X-API-Key': 'your_api_key'
  }
});

const data = await response.json();
console.log(data.tenant.usage); // { tokens_used, token_quota, cost_used, cost_limit }
```

---

## Development

### Installation

```bash
cd dashboard
npm install
# or
pnpm install
```

### Running Locally

```bash
npm run dev
```

The dashboard will be available at `http://localhost:5173` (Vite default).

### Building for Production

```bash
npm run build
```

This creates an optimized build in the `dist/` directory.

### Type Checking

```bash
npm run check
```

Runs TypeScript compiler to check for type errors.

### Code Formatting

```bash
npm run format
```

Runs Prettier to format all files.

---

## Deployment

### Docker

The dashboard is included in the main `docker-compose.yml`:

```yaml
services:
  leo-optima:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=your_key
      - LEO_API_KEY=your_secret_key
    volumes:
      - leo_storage:/app/leo_storage
      - redis_data:/data

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "3000:3000"
    depends_on:
      - leo-optima
    environment:
      - VITE_API_URL=http://leo-optima:8000
```

### Standalone Deployment

To deploy the dashboard independently:

```bash
npm run build
npm run start
```

The server will listen on port 3000 by default.

---

## Component Library

### MetricCard

Displays a single KPI with icon, value, trend, and description.

```tsx
<MetricCard
  title="Total Cost Saved"
  value="$1,247.50"
  icon={<DollarSign className="w-6 h-6" />}
  trend={{ value: 24, isPositive: true }}
  description="This month"
  color="green"
/>
```

**Props:**
- `title`: Metric label
- `value`: Main metric value
- `unit`: Optional unit (e.g., "M" for millions)
- `icon`: React component for icon
- `trend`: Optional trend object with `value` and `isPositive`
- `description`: Optional helper text
- `color`: Color variant (blue, green, amber, red)

### DashboardLayout

Wraps page content with sidebar and top bar.

```tsx
<DashboardLayout currentTenant={tenant}>
  {/* Page content */}
</DashboardLayout>
```

**Props:**
- `children`: Page content
- `currentTenant`: Optional tenant object with `id`, `name`, `tier`

---

## Customization

### Adding New Pages

1. Create a new file in `client/src/pages/`
2. Add the route to `client/src/App.tsx`
3. Update the sidebar navigation in `client/src/components/Sidebar.tsx`

Example:

```tsx
// client/src/pages/Analytics.tsx
export default function Analytics() {
  return (
    <DashboardLayout>
      <h1>Analytics</h1>
      {/* Your content */}
    </DashboardLayout>
  );
}
```

### Changing Colors

Edit the CSS variables in `client/src/index.css`:

```css
:root {
  --primary: #3B82F6;  /* Change primary color */
  --accent: #10B981;   /* Change accent color */
  /* ... more variables ... */
}
```

### Adding Charts

The dashboard uses Recharts. Example of adding a new chart:

```tsx
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

<ResponsiveContainer width="100%" height={300}>
  <BarChart data={data}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="name" />
    <YAxis />
    <Tooltip />
    <Legend />
    <Bar dataKey="value" fill="#3B82F6" />
  </BarChart>
</ResponsiveContainer>
```

---

## Performance Optimization

- **Code Splitting**: Automatic via Vite
- **Image Optimization**: Use WebP format with fallbacks
- **Lazy Loading**: Route-based code splitting with React
- **Caching**: Browser caching with cache headers
- **Compression**: Gzip compression on the server

---

## Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions
- Mobile browsers: iOS Safari 12+, Chrome Android 90+

---

## Troubleshooting

### Dashboard won't load

1. Check that the LEO Optima backend is running on port 8000
2. Verify the API key is correct in the environment
3. Check browser console for CORS errors
4. Ensure `VITE_API_URL` environment variable is set

### Charts not displaying

1. Verify Recharts is installed: `npm list recharts`
2. Check that chart data is in the correct format
3. Ensure ResponsiveContainer has a parent with defined height

### Styling issues

1. Clear browser cache (Ctrl+Shift+Delete)
2. Rebuild CSS: `npm run build`
3. Check that Tailwind CSS is properly configured in `tailwind.config.js`

---

## Future Enhancements

- Real-time WebSocket updates for live metrics
- Custom dashboard widgets
- Export reports to PDF
- Dark mode toggle
- Multi-language support
- Advanced filtering and search
- Webhook management UI
- Audit logs viewer

---

## Contributing

To contribute to the dashboard:

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes
3. Test locally: `npm run dev`
4. Commit with clear messages: `git commit -m "feat: add new feature"`
5. Push and create a pull request

---

## License

MIT - See LICENSE file for details

---

## Support

For issues or questions:
- Open an issue on GitHub
- Check the main README.md
- Review API_DOCUMENTATION.md
