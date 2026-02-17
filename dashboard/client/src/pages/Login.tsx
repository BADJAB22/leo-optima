import { useState } from 'react';
import { useLocation } from 'wouter';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { setApiKey } from '@/lib/auth';
import { Key, Shield } from 'lucide-react';

export default function Login() {
  const [key, setKey] = useState('');
  const [, setLocation] = useLocation();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (key.trim()) {
      setApiKey(key.trim());
      setLocation('/');
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <Card className="w-full max-w-md p-8 space-y-6">
        <div className="text-center space-y-2">
          <div className="flex justify-center">
            <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center">
              <Shield className="text-primary-foreground w-8 h-8" />
            </div>
          </div>
          <h1 className="text-2xl font-bold">LEO Optima</h1>
          <p className="text-muted-foreground text-[11px] uppercase tracking-widest font-bold">Kadropic Analytics Platform</p>
        </div>

        <form onSubmit={handleLogin} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
              LEO API Key
            </label>
            <div className="relative">
              <Key className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
              <Input
                type="password"
                placeholder="leo_..."
                className="pl-10"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                required
              />
            </div>
          </div>
          <Button type="submit" className="w-full">
            Access Dashboard
          </Button>
        </form>

        <div className="text-center text-xs text-muted-foreground pt-4 border-t border-border/50">
          <p className="font-medium">Bader Jamal</p>
          <p className="text-[10px]">Founder, Kadropic Labs</p>
        </div>
      </Card>
    </div>
  );
}
