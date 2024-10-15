import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from typing import List, Optional, Dict
from topos.utilities.utils import generate_deci_code

class GroupManagerPostgres:
    def __init__(self, db_params: Dict[str, str]):
        self.db_params = db_params

    def _get_connection(self):
        return psycopg2.connect(**self.db_params)

    def create_group(self, group_name: str) -> str:
        group_id = generate_deci_code(6)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('INSERT INTO groups (group_id, group_name) VALUES (%s, %s)', (group_id, group_name))
        return group_id

    def create_user(self, user_id: str, username: str) -> str:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('INSERT INTO users (user_id, username) VALUES (%s, %s)', (user_id, username))
        return user_id

    def add_user_to_group(self, user_id: str, group_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute('INSERT INTO user_groups (user_id, group_id) VALUES (%s, %s)', (user_id, group_id))
            return True
        except psycopg2.IntegrityError:
            return False

    def remove_user_from_group(self, user_id: str, group_id: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM user_groups WHERE user_id = %s AND group_id = %s', (user_id, group_id))
                return cur.rowcount > 0

    def get_user_groups(self, user_id: str) -> List[dict]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute('''
                    SELECT g.group_id, g.group_name
                    FROM groups g
                    JOIN user_groups ug ON g.group_id = ug.group_id
                    WHERE ug.user_id = %s
                ''', (user_id,))
                return [dict(row) for row in cur.fetchall()]

    def get_group_users(self, group_id: str) -> List[dict]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute('''
                    SELECT u.user_id, u.username
                    FROM users u
                    JOIN user_groups ug ON u.user_id = ug.user_id
                    WHERE ug.group_id = %s
                ''', (group_id,))
                return [dict(row) for row in cur.fetchall()]

    def get_group_by_id(self, group_id: str) -> Optional[dict]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute('SELECT group_id, group_name FROM groups WHERE group_id = %s', (group_id,))
                result = cur.fetchone()
                return dict(result) if result else None

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute('SELECT user_id, username FROM users WHERE user_id = %s', (user_id,))
                result = cur.fetchone()
                return dict(result) if result else None

    def get_group_by_name(self, group_name: str) -> Optional[dict]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute('SELECT group_id, group_name FROM groups WHERE group_name = %s', (group_name,))
                result = cur.fetchone()
                return dict(result) if result else None

    def get_user_by_username(self, username: str) -> Optional[dict]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute('SELECT user_id, username FROM users WHERE username = %s', (username,))
                result = cur.fetchone()
                return dict(result) if result else None

    def delete_group(self, group_id: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM user_groups WHERE group_id = %s', (group_id,))
                cur.execute('DELETE FROM groups WHERE group_id = %s', (group_id,))
                return cur.rowcount > 0

    def delete_user(self, user_id: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM user_groups WHERE user_id = %s', (user_id,))
                cur.execute('DELETE FROM users WHERE user_id = %s', (user_id,))
                return cur.rowcount > 0

    def get_user_last_seen_online(self, user_id: str) -> Optional[str]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT last_seen_online FROM users WHERE user_id = %s', (user_id,))
                result = cur.fetchone()
                return result[0].isoformat() if result else None

    def set_user_last_seen_online(self, user_id: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('UPDATE users SET last_seen_online = %s WHERE user_id = %s', (datetime.now(), user_id))
                return cur.rowcount > 0
