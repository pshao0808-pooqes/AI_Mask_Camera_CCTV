import sqlite3
from datetime import datetime
import json

class TrafficDatabase:
    def __init__(self, db_path="traffic_data.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cctv_id INTEGER,
                cctv_name TEXT,
                timestamp DATETIME,
                vehicle_count INTEGER,
                traffic_level TEXT,
                avg_speed REAL,
                lat REAL,
                lon REAL,
                raw_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_traffic_data(self, cctv_id, cctv_name, traffic_data, lat, lon):
        """트래픽 데이터 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO traffic_logs 
            (cctv_id, cctv_name, timestamp, vehicle_count, traffic_level, avg_speed, lat, lon, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cctv_id,
            cctv_name,
            datetime.now(),
            traffic_data['vehicle_count'],
            traffic_data['traffic_level'],
            traffic_data['avg_speed'],
            lat,
            lon,
            json.dumps(traffic_data)
        ))
        
        conn.commit()
        conn.close()
    
    def get_traffic_history(self, cctv_id, hours=24):
        """트래픽 히스토리 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM traffic_logs 
            WHERE cctv_id = ? AND timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours), (cctv_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
