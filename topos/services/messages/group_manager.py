from datetime import datetime
import sqlite3
import uuid
from typing import List, Optional, Dict
from topos.utilities.utils import generate_deci_code

class GroupManagerSQLite:
    def __init__(self, db_file: str = '../db/user.db'):
        self.db_file = db_file

        # Initialize empty caches
        self.groups_cache: Dict[str, Dict] = {}  # group_id -> group_info
        self.users_cache: Dict[str, Dict] = {}  # user_id -> user_info
        self.user_groups_cache: Dict[str, List[str]] = {}  # user_id -> list of group_ids
        self.group_users_cache: Dict[str, List[str]] = {}  # group_id -> list of user_ids

    def _get_group_from_db(self, group_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT group_id, group_name FROM groups WHERE group_id = ?', (group_id,))
            result = cursor.fetchone()
            print(result)
            if result:
                return {"group_id": result[0], "group_name": result[1]}
        return None

    def _get_user_from_db(self, user_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, username FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            if result:
                return {"user_id": result[0], "username": result[1]}
        return None

    def _get_user_groups_from_db(self, user_id: str) -> List[str]:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT group_id FROM user_groups WHERE user_id = ?', (user_id,))
            return [row[0] for row in cursor.fetchall()]

    def _get_group_users_from_db(self, group_id: str) -> List[str]:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id FROM user_groups WHERE group_id = ?', (group_id,))
            return [row[0] for row in cursor.fetchall()]

    def create_group(self, group_name: str) -> str:
        group_id = generate_deci_code(6)
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO groups (group_id, group_name) VALUES (?, ?)', (group_id, group_name))
            conn.commit()

        # Update cache
        self.groups_cache[group_id] = {"group_id": group_id, "group_name": group_name}
        self.group_users_cache[group_id] = []

        return group_id

    def create_user(self, user_id:str,username: str,) -> str:

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (user_id, username) VALUES (?, ?)', (user_id, username))
            conn.commit()

        # Update cache
        self.users_cache[user_id] = {"user_id": user_id, "username": username}
        self.user_groups_cache[user_id] = []

        return user_id

    def add_user_to_group(self, user_id: str, group_id: str) -> bool:
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO user_groups (user_id, group_id) VALUES (?, ?)', (user_id, group_id))
                conn.commit()

            # Update cache if the entries exist
            if user_id in self.user_groups_cache:
                self.user_groups_cache[user_id].append(group_id)
            if group_id in self.group_users_cache:
                self.group_users_cache[group_id].append(user_id)

            return True
        except sqlite3.IntegrityError:
            return False  # User already in group or user/group doesn't exist

    def remove_user_from_group(self, user_id: str, group_id: str) -> bool:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM user_groups WHERE user_id = ? AND group_id = ?', (user_id, group_id))
            conn.commit()
            if cursor.rowcount > 0:
                # Update cache if the entries exist
                if user_id in self.user_groups_cache:
                    self.user_groups_cache[user_id].remove(group_id)
                if group_id in self.group_users_cache:
                    self.group_users_cache[group_id].remove(user_id)
                return True
            return False

    def get_user_groups(self, user_id: str) -> List[dict]:
        if user_id not in self.user_groups_cache:
            self.user_groups_cache[user_id] = self._get_user_groups_from_db(user_id)

        return [self.get_group_by_id(group_id) for group_id in self.user_groups_cache[user_id]]

    def get_group_users(self, group_id: str) -> List[dict]:
        if group_id not in self.group_users_cache:
            self.group_users_cache[group_id] = self._get_group_users_from_db(group_id)

        return [self.get_user_by_id(user_id) for user_id in self.group_users_cache[group_id]]

    def get_group_by_id(self, group_id: str) -> Optional[dict]:
        if group_id not in self.groups_cache:
            group = self._get_group_from_db(group_id)
            if group:
                self.groups_cache[group_id] = group
            else:
                return None
        return self.groups_cache[group_id]

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        if user_id not in self.users_cache:
            user = self._get_user_from_db(user_id)
            if user:
                self.users_cache[user_id] = user
            else:
                return None
        return self.users_cache[user_id]

    def get_group_by_name(self, group_name: str) -> Optional[dict]:
        # This operation requires a full DB scan if not in cache
        for group in self.groups_cache.values():
            if group['group_name'] == group_name:
                return group

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT group_id, group_name FROM groups WHERE group_name = ?', (group_name,))
            result = cursor.fetchone()
            if result:
                group = {"group_id": result[0], "group_name": result[1]}
                self.groups_cache[group['group_id']] = group
                return group
        return None

    def get_user_by_username(self, username: str) -> Optional[dict]:
        # This operation requires a full DB scan if not in cache
        for user in self.users_cache.values():
            if user['username'] == username:
                return user

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, username FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            if result:
                user = {"user_id": result[0], "username": result[1]}
                self.users_cache[user['user_id']] = user
                return user
        return None

    def delete_group(self, group_id: str) -> bool:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM user_groups WHERE group_id = ?', (group_id,))
            cursor.execute('DELETE FROM groups WHERE group_id = ?', (group_id,))
            conn.commit()
            if cursor.rowcount > 0:
                # Update cache
                self.groups_cache.pop(group_id, None)
                self.group_users_cache.pop(group_id, None)
                for user_groups in self.user_groups_cache.values():
                    if group_id in user_groups:
                        user_groups.remove(group_id)
                return True
            return False

    def delete_user(self, user_id: str) -> bool:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM user_groups WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
            conn.commit()
            if cursor.rowcount > 0:
                # Update cache
                self.users_cache.pop(user_id, None)
                self.user_groups_cache.pop(user_id, None)
                for group_users in self.group_users_cache.values():
                    if user_id in group_users:
                        group_users.remove(user_id)
                return True
            return False
    def get_user_last_seen_online(self, user_id: str) -> str:
            """
            Get the last_seen_online timestamp for a given user_id.

            :param user_id: The ID of the user
            :return: The last seen timestamp as a string, or None if user not found
            """
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                    SELECT last_seen_online
                    FROM users
                    WHERE user_id = ?
                    ''', (user_id,))

                    result = cursor.fetchone()
                    if result:
                        return result[0]
                    else:
                        print(f"User with ID {user_id} not found.")
                        return None
                except sqlite3.Error as e:
                    print(f"An error occurred: {e}")
                    return None

    def set_user_last_seen_online(self, user_id: str) -> bool:
            """
            Set the last_seen_online timestamp for a given user_id to the current time.

            :param user_id: The ID of the user
            :return: True if successful, False if user not found
            """
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                    UPDATE users
                    SET last_seen_online = ?
                    WHERE user_id = ?
                    ''', (datetime.now().replace(microsecond=0), user_id))

                    if cursor.rowcount == 0:
                        print(f"User with ID {user_id} not found.")
                        return False

                    return True
                except sqlite3.Error as e:
                    print(f"An error occurred: {e}")
                    return False
