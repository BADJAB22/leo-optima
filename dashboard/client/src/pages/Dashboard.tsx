import { useState, useEffect } from 'react';
import { DollarSign, Zap, TrendingUp, Activity } from 'lucide-react';
import DashboardLayout from '@/components/DashboardLayout';
import MetricCard from '@/components/MetricCard';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Card } from '@/components/ui/card';
import { getApiKey, getApiBaseUrl } from '@/lib/auth';
import axios from 'axios';

interface AnalyticsData {
  tenant: {
    id: string;
    name: string;
    usage: {
      tokens_used: number;
      token_quota: number;
      cost_used: number;
      cost_limit: number;
    }
  };
  optimization_metrics: {
    total_requests: number;
    cache_hits: number;
    cache_hit_rate: number;
    tokens_saved: number;
    cost_saved: number;
    average_confidence: number;
  };
}

export default function Dashboard() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      const apiKey = getApiKey() || 'leo_admin_secret_key'; // Fallback for initial view
      try {
        setLoading(true);
        const response = await axios.get(`${getApiBaseUrl()}/v1/analytics`, {
          headers: { 'X-API-Key': apiKey }
        });
        setData(response.data);
        setError(null);
      } catch (err: any) {
        console.error('Failed to fetch analytics:', err);
        setError(err.response?.data?.detail || 'Failed to connect to LEO Optima API');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </DashboardLayout>
    );
  }

  if (error && !data) {
    return (
      <DashboardLayout>
        <div className="p-6 bg-destructive/10 border border-destructive rounded-lg text-destructive">
          <h2 className="text-lg font-bold">Error</h2>
          <p>{error}</p>
          <p className="mt-2 text-sm">Make sure the LEO Optima server is running and your API key is correct.</p>
        </div>
      </DashboardLayout>
    );
  }

  const metrics = data?.optimization_metrics || {
    cost_saved: 0,
    tokens_saved: 0,
    cache_hit_rate: 0,
    total_requests: 0
  };

  const tenant = data?.tenant || {
    name: 'Loading...'
  };

  const routeDistribution = [
    { name: 'Cache Hits', value: Math.round(metrics.cache_hit_rate * 100), color: '#10B981' },
    { name: 'API Requests', value: 100 - Math.round(metrics.cache_hit_rate * 100), color: '#3B82F6' },
  ];

  return (
    <DashboardLayout currentTenant={{ id: tenant.id, name: tenant.name }}>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground mb-2">Community Dashboard</h1>
          <p className="text-muted-foreground">Real-time performance metrics for {tenant.name}.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Total Cost Saved"
            value={`$${metrics.cost_saved.toFixed(2)}`}
            icon={<DollarSign className="w-6 h-6" />}
            description="Lifetime savings"
            color="green"
          />
          <MetricCard
            title="Tokens Optimized"
            value={(metrics.tokens_saved / 1000).toFixed(1)}
            unit="K"
            icon={<Zap className="w-6 h-6" />}
            description="Tokens reduced"
            color="blue"
          />
          <MetricCard
            title="Cache Hit Rate"
            value={`${(metrics.cache_hit_rate * 100).toFixed(1)}%`}
            icon={<TrendingUp className="w-6 h-6" />}
            description="Optimization efficiency"
            color="green"
          />
          <MetricCard
            title="Requests Processed"
            value={metrics.total_requests.toLocaleString()}
            icon={<Activity className="w-6 h-6" />}
            description="Total processed"
            color="amber"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 p-6">
            <h2 className="text-lg font-bold text-foreground mb-4">Quota Usage</h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium">Token Quota</span>
                  <span className="text-sm text-muted-foreground">
                    {data?.tenant.usage.tokens_used.toLocaleString()} / {data?.tenant.usage.token_quota.toLocaleString()}
                  </span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2.5">
                  <div 
                    className="bg-primary h-2.5 rounded-full" 
                    style={{ width: `${Math.min(100, (data?.tenant.usage.tokens_used || 0) / (data?.tenant.usage.token_quota || 1) * 100)}%` }}
                  ></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium">Cost Limit</span>
                  <span className="text-sm text-muted-foreground">
                    ${data?.tenant.usage.cost_used.toFixed(2)} / ${data?.tenant.usage.cost_limit.toFixed(2)}
                  </span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2.5">
                  <div 
                    className="bg-amber-500 h-2.5 rounded-full" 
                    style={{ width: `${Math.min(100, (data?.tenant.usage.cost_used || 0) / (data?.tenant.usage.cost_limit || 1) * 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h2 className="text-lg font-bold text-foreground mb-4">Optimization Split</h2>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={routeDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
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
      </div>
    </DashboardLayout>
  );
}
