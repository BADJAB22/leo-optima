import { useState } from 'react';
import { Menu, X, LogOut, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Sidebar from './Sidebar';

interface DashboardLayoutProps {
  children: React.ReactNode;
  currentTenant?: {
    id: string;
    name: string;
  };
}

export default function DashboardLayout({ children, currentTenant }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Sidebar */}
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <div className="h-16 bg-card border-b border-border flex items-center justify-between px-6 shadow-sm">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
            <h1 className="text-xl font-bold text-foreground hidden sm:block">LEO Optima Community</h1>
          </div>

          {/* Tenant Info & Actions */}
          <div className="flex items-center gap-4">
            {currentTenant && (
              <div className="hidden md:flex items-center gap-2">
                <div className="text-right">
                  <p className="text-sm font-medium text-foreground">{currentTenant.name}</p>
                  <p className="text-xs text-muted-foreground capitalize">Community Edition</p>
                </div>
              </div>
            )}

            <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
              <Settings className="w-5 h-5" />
            </Button>

            <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
              <LogOut className="w-5 h-5" />
            </Button>
          </div>
        </div>

        {/* Page Content */}
        <div className="flex-1 overflow-auto">
          <div className="p-6">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
