import { BarChart3, Settings, Key, Users, TrendingDown, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Link } from 'wouter';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const navItems = [
  { icon: Home, label: 'Dashboard', href: '/', active: true },
  { icon: TrendingDown, label: 'Analytics', href: '/analytics', active: false },
  { icon: BarChart3, label: 'Metrics', href: '/metrics', active: false },
  { icon: Users, label: 'Tenants', href: '/tenants', active: false },
  { icon: Key, label: 'API Keys', href: '/api-keys', active: false },
  { icon: Settings, label: 'Settings', href: '/settings', active: false },
];

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div
        className={`
          fixed lg:static inset-y-0 left-0 z-40 w-64 bg-sidebar border-r border-sidebar-border
          transform transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        <div className="h-full flex flex-col">
          {/* Logo Section */}
          <div className="h-16 flex items-center justify-center border-b border-sidebar-border">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-sidebar-primary rounded-lg flex items-center justify-center">
                <span className="text-sidebar-primary-foreground font-bold text-sm">L</span>
              </div>
              <span className="font-bold text-sidebar-foreground hidden sm:inline">LEO</span>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link key={item.href} href={item.href}>
                  <a
                    onClick={onClose}
                    className={`
                      w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors
                      ${item.active
                        ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                        : 'text-sidebar-foreground hover:bg-sidebar-accent'
                      }
                    `}
                  >
                    <Icon className="w-5 h-5 flex-shrink-0" />
                    <span className="text-sm font-medium">{item.label}</span>
                  </a>
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="border-t border-sidebar-border p-4 space-y-3">
            <div className="text-xs text-sidebar-foreground/60">
              <p>LEO Optima v2.0</p>
              <p>Multi-Tenant Edition</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
