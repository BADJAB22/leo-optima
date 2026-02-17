import { useState, useEffect } from 'react';
import { DollarSign, Zap, TrendingUp, Activity } from 'lucide-react';
import DashboardLayout from '@/components/DashboardLayout';
import MetricCard from '@/components/MetricCard';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Card } from '@/components/ui/card';

// Mock data - in production, this would come from the API
const mockMetrics = {
  costSaved: 1247.50,
  tokensOptimized: 2847500,
  cacheHitRate: 0.42,
  requestsProcessed: 12847,
};

const mockChartData = [
  { time: '00:00', cost: 0, tokens: 0, cache: 0 },
  { time: '04:00', cost: 45, tokens: 125000, cache: 15 },
  { time: '08:00', cost: 120, tokens: 450000, cache: 28 },
  { time: '12:00', cost: 280, tokens: 890000, cache: 38 },
  { time: '16:00', cost: 650, tokens: 1850000, cache: 42 },
  { time: '20:00', cost: 1100, tokens: 2500000, cache: 45 },
  { time: '24:00', cost: 1247.50, tokens: 2847500, cache: 42 },
];

const routeDistribution = [
  { name: 'Cache Hits', value: 42, color: '#10B981' },
  { name: 'Fast Route', value: 38, color: '#3B82F6' },
  { name: 'Consensus', value: 20, color: '#F59E0B' },
];

const tenantData = [
  { name: 'Tenant A', requests: 3500, savings: 350, cacheHit: 45 },
  { name: 'Tenant B', requests: 2800, savings: 280, cacheHit: 38 },
  { name: 'Tenant C', requests: 2100, savings: 210, cacheHit: 35 },
  { name: 'Tenant D', requests: 1900, savings: 190, cacheHit: 32 },
  { name: 'Tenant E', requests: 1547, savings: 217, cacheHit: 42 },
];

export default function Dashboard() {
  const [metrics, setMetrics] = useState(mockMetrics);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Simulate fetching data from the API
    setLoading(true);
    const timer = setTimeout(() => {
      setLoading(false);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  const currentTenant = {
    id: 'tenant-001',
    name: 'Acme Corp',
    tier: 'pro',
  };

  return (
    <DashboardLayout currentTenant={currentTenant}>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground mb-2">Dashboard</h1>
          <p className="text-muted-foreground">Welcome back! Here's your LEO Optima performance overview.</p>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Total Cost Saved"
            value={`$${metrics.costSaved.toFixed(2)}`}
            icon={<DollarSign className="w-6 h-6" />}
            trend={{ value: 24, isPositive: true }}
            description="This month"
            color="green"
          />
          <MetricCard
            title="Tokens Optimized"
            value={(metrics.tokensOptimized / 1000000).toFixed(1)}
            unit="M"
            icon={<Zap className="w-6 h-6" />}
            trend={{ value: 18, isPositive: true }}
            description="Tokens reduced"
            color="blue"
          />
          <MetricCard
            title="Cache Hit Rate"
            value={`${(metrics.cacheHitRate * 100).toFixed(1)}%`}
            icon={<TrendingUp className="w-6 h-6" />}
            trend={{ value: 12, isPositive: true }}
            description="Avg. this week"
            color="green"
          />
          <MetricCard
            title="Requests Processed"
            value={metrics.requestsProcessed.toLocaleString()}
            icon={<Activity className="w-6 h-6" />}
            trend={{ value: 8, isPositive: true }}
            description="Total today"
            color="amber"
          />
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Cost & Token Savings Chart */}
          <Card className="lg:col-span-2 p-6">
            <h2 className="text-lg font-bold text-foreground mb-4">Cost & Token Savings Over Time</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mockChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="time" stroke="#64748B" />
                <YAxis stroke="#64748B" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E2E8F0',
                    borderRadius: '0.5rem',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="cost"
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={false}
                  name="Cost Saved ($)"
                />
                <Line
                  type="monotone"
                  dataKey="tokens"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={false}
                  name="Tokens Saved (K)"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Route Distribution */}
          <Card className="p-6">
            <h2 className="text-lg font-bold text-foreground mb-4">Request Routes</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={routeDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {routeDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-4 space-y-2">
              {routeDistribution.map((item) => (
                <div key={item.name} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="text-muted-foreground">{item.name}</span>
                  </div>
                  <span className="font-semibold text-foreground">{item.value}%</span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Tenant Performance */}
        <Card className="p-6">
          <h2 className="text-lg font-bold text-foreground mb-4">Multi-Tenant Performance</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Tenant</th>
                  <th className="text-right py-3 px-4 font-semibold text-foreground">Requests</th>
                  <th className="text-right py-3 px-4 font-semibold text-foreground">Cost Saved</th>
                  <th className="text-right py-3 px-4 font-semibold text-foreground">Cache Hit %</th>
                </tr>
              </thead>
              <tbody>
                {tenantData.map((tenant) => (
                  <tr key={tenant.name} className="border-b border-border hover:bg-card/50 transition-colors">
                    <td className="py-3 px-4 text-foreground font-medium">{tenant.name}</td>
                    <td className="text-right py-3 px-4 text-muted-foreground">{tenant.requests.toLocaleString()}</td>
                    <td className="text-right py-3 px-4 text-green-600 font-semibold">${tenant.savings}</td>
                    <td className="text-right py-3 px-4 text-foreground">{tenant.cacheHit}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </DashboardLayout>
  );
}
