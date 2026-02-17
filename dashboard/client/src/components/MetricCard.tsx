import { ReactNode } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon: ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  description?: string;
  color?: 'blue' | 'green' | 'amber' | 'red';
}

const colorClasses = {
  blue: 'text-blue-600 bg-blue-50 dark:bg-blue-900/20',
  green: 'text-green-600 bg-green-50 dark:bg-green-900/20',
  amber: 'text-amber-600 bg-amber-50 dark:bg-amber-900/20',
  red: 'text-red-600 bg-red-50 dark:bg-red-900/20',
};

export default function MetricCard({
  title,
  value,
  unit,
  icon,
  trend,
  description,
  color = 'blue',
}: MetricCardProps) {
  return (
    <div className="bg-card rounded-lg border border-border p-6 shadow-sm hover:shadow-md transition-shadow duration-200">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          {icon}
        </div>
        {trend && (
          <div className={`flex items-center gap-1 text-sm font-medium ${trend.isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {trend.isPositive ? (
              <TrendingUp className="w-4 h-4" />
            ) : (
              <TrendingDown className="w-4 h-4" />
            )}
            {trend.value}%
          </div>
        )}
      </div>

      {/* Content */}
      <div className="space-y-2">
        <p className="text-sm text-muted-foreground font-medium">{title}</p>
        <div className="flex items-baseline gap-2">
          <p className="text-3xl font-bold text-foreground">{value}</p>
          {unit && <p className="text-sm text-muted-foreground">{unit}</p>}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground mt-3">{description}</p>
        )}
      </div>
    </div>
  );
}
