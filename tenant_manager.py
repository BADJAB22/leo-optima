import sqlite3
import hashlib
import secrets
import os
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Tenant:
    id: str
    name: str
    api_key_hash: str
    token_quota: int
    tokens_used: int
    cost_limit: float
    cost_used: float
    is_active: bool
    created_at: str

class TenantManager:
    """Professional Identity and Quota Manager for Multi-Tenant LEO Optima"""
    
    def __init__(self, db_path: str = "leo_storage/identity.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists and has api_key column (to migrate if needed)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tenants'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            cursor.execute("PRAGMA table_info(tenants)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'api_key' in columns:
                cursor.execute("ALTER TABLE tenants RENAME COLUMN api_key TO api_key_hash")
            if 'tier' in columns:
                # We don't need tier in open source version
                pass
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    api_key_hash TEXT UNIQUE NOT NULL,
                    token_quota INTEGER DEFAULT 1000000,
                    tokens_used INTEGER DEFAULT 0,
                    cost_limit REAL DEFAULT 100.0,
                    cost_used REAL DEFAULT 0.0,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
        # Create a default admin tenant if none exists
        cursor.execute("SELECT COUNT(*) FROM tenants")
        if cursor.fetchone()[0] == 0:
            admin_key = os.getenv("LEO_API_KEY", "leo_admin_secret_key")
            self.create_tenant("Admin", api_key=admin_key, tier="enterprise", token_quota=10**9, cost_limit=10**6)
        
        conn.commit()
        conn.close()

    def _hash_key(self, api_key: str) -> str:
        """Hash the API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_tenant(self, name: str, api_key: Optional[str] = None, 
                      token_quota: int = 1000000, cost_limit: float = 100.0) -> Dict:
        if not api_key:
            api_key = f"leo_{secrets.token_urlsafe(32)}"
        
        key_hash = self._hash_key(api_key)
        tenant_id = key_hash[:12]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO tenants (id, name, api_key_hash, token_quota, cost_limit)
                VALUES (?, ?, ?, ?, ?)
            ''', (tenant_id, name, key_hash, token_quota, cost_limit))
            conn.commit()
            return {"id": tenant_id, "api_key": api_key, "name": name}
        except sqlite3.IntegrityError:
            return {"error": "API Key already exists"}
        finally:
            conn.close()

    def get_tenant_by_key(self, api_key: str) -> Optional[Tenant]:
        key_hash = self._hash_key(api_key)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tenants WHERE api_key_hash = ? AND is_active = 1", (key_hash,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Map the row to Tenant dataclass
            data = dict(row)
            return Tenant(**data)
        return None

    def update_usage(self, tenant_id: str, tokens: int, cost: float):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE tenants 
            SET tokens_used = tokens_used + ?, cost_used = cost_used + ?
            WHERE id = ?
        ''', (tokens, cost, tenant_id))
        conn.commit()
        conn.close()

    def check_quota(self, tenant: Tenant) -> bool:
        """Check if tenant has exceeded their quota or cost limit"""
        if tenant.tokens_used >= tenant.token_quota:
            return False
        if tenant.cost_used >= tenant.cost_limit:
            return False
        return True

    def list_tenants(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, tokens_used, token_quota, cost_used, cost_limit, is_active FROM tenants")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
