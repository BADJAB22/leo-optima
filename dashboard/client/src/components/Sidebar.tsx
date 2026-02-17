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
          <div className="h-20 flex items-center px-6 border-b border-sidebar-border">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-sidebar-primary rounded-xl flex items-center justify-center shadow-lg shadow-sidebar-primary/20">
                <TrendingDown className="text-sidebar-primary-foreground w-5 h-5" />
              </div>
              <div className="flex flex-col">
                <span className="font-bold text-sidebar-foreground text-sm leading-none">LEO Optima</span>
                <span className="text-[10px] text-sidebar-foreground/50 mt-1 uppercase tracking-wider font-medium">By BADJAB</span>
              </div>
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
          <div className="border-t border-sidebar-border p-6">
            <div className="bg-sidebar-accent/50 rounded-xl p-4 border border-sidebar-border">
              <p className="text-xs font-bold text-sidebar-foreground mb-1">Community Hub</p>
              <p className="text-[10px] text-sidebar-foreground/60 mb-3">Built for the AI Community</p>
              <div className="flex flex-col gap-2">
                <Button variant="outline" size="sm" className="w-full text-[10px] h-8 bg-sidebar-background border-sidebar-border hover:bg-sidebar-accent" asChild>
                  <a href="https://github.com/BADJAB22/leo-optima" target="_blank" rel="noreferrer">
                    Star on GitHub
                  </a>
                </Button>
                <a href="https://twitter.com/BADJAB22" target="_blank" rel="noreferrer" className="text-[10px] text-sidebar-foreground/40 hover:text-sidebar-foreground text-center transition-colors">
                  Follow @BADJAB22
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
