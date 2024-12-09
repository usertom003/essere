import sqlite3
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MovementDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.initialize_database()
        
    def initialize_database(self):
        """Create database tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Sessions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    subject_id TEXT,
                    notes TEXT
                )
                ''')
                
                # Pose data table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS pose_data (
                    pose_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    timestamp TIMESTAMP,
                    landmark_data TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                ''')
                
                # Movement metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS movement_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pose_id INTEGER,
                    smoothness REAL,
                    complexity REAL,
                    periodicity REAL,
                    symmetry REAL,
                    coordination REAL,
                    energy REAL,
                    variability TEXT,
                    phase_coherence REAL,
                    FOREIGN KEY (pose_id) REFERENCES pose_data(pose_id)
                )
                ''')
                
                # Movement patterns table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS movement_patterns (
                    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pose_id INTEGER,
                    pattern_name TEXT,
                    confidence REAL,
                    frequency REAL,
                    duration REAL,
                    intensity REAL,
                    FOREIGN KEY (pose_id) REFERENCES pose_data(pose_id)
                )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
            
    def start_session(self, subject_id: str, notes: str = "") -> int:
        """Start a new recording session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO sessions (start_time, subject_id, notes)
                VALUES (?, ?, ?)
                ''', (datetime.now().isoformat(), subject_id, notes))
                
                session_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Started new session {session_id} for subject {subject_id}")
                return session_id
                
        except Exception as e:
            logger.error(f"Error starting session: {str(e)}")
            raise
            
    def end_session(self, session_id: int):
        """End a recording session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE sessions
                SET end_time = ?
                WHERE session_id = ?
                ''', (datetime.now().isoformat(), session_id))
                
                conn.commit()
                logger.info(f"Ended session {session_id}")
                
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            raise
            
    def store_pose_data(self, session_id: int, landmarks: List[Any]) -> int:
        """Store pose landmarks data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert landmarks to serializable format
                landmark_data = [
                    {
                        'x': float(l.x),
                        'y': float(l.y),
                        'z': float(l.z),
                        'visibility': float(l.visibility)
                    }
                    for l in landmarks
                ]
                
                cursor.execute('''
                INSERT INTO pose_data (session_id, timestamp, landmark_data)
                VALUES (?, ?, ?)
                ''', (
                    session_id,
                    datetime.now().isoformat(),
                    json.dumps(landmark_data)
                ))
                
                pose_id = cursor.lastrowid
                conn.commit()
                return pose_id
                
        except Exception as e:
            logger.error(f"Error storing pose data: {str(e)}")
            raise
            
    def store_movement_metrics(self, pose_id: int, metrics: Dict):
        """Store movement analysis metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO movement_metrics (
                    pose_id, smoothness, complexity, periodicity,
                    symmetry, coordination, energy, variability,
                    phase_coherence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pose_id,
                    metrics['smoothness'],
                    metrics['complexity'],
                    metrics['periodicity'],
                    metrics['symmetry'],
                    metrics['coordination'],
                    metrics['energy'],
                    json.dumps(metrics['variability']),
                    metrics['phase_coherence']
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing movement metrics: {str(e)}")
            raise
            
    def store_movement_patterns(self, pose_id: int, patterns: List[Dict]):
        """Store detected movement patterns"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for pattern in patterns:
                    cursor.execute('''
                    INSERT INTO movement_patterns (
                        pose_id, pattern_name, confidence,
                        frequency, duration, intensity
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        pose_id,
                        pattern['name'],
                        pattern['confidence'],
                        pattern['frequency'],
                        pattern['duration'],
                        pattern['intensity']
                    ))
                    
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing movement patterns: {str(e)}")
            raise
            
    def get_session_data(self, session_id: int) -> Dict:
        """Get all data for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create DataFrames
                session_df = pd.read_sql_query('''
                SELECT * FROM sessions WHERE session_id = ?
                ''', conn, params=(session_id,))
                
                pose_df = pd.read_sql_query('''
                SELECT * FROM pose_data WHERE session_id = ?
                ''', conn, params=(session_id,))
                
                metrics_df = pd.read_sql_query('''
                SELECT mm.* FROM movement_metrics mm
                JOIN pose_data pd ON mm.pose_id = pd.pose_id
                WHERE pd.session_id = ?
                ''', conn, params=(session_id,))
                
                patterns_df = pd.read_sql_query('''
                SELECT mp.* FROM movement_patterns mp
                JOIN pose_data pd ON mp.pose_id = pd.pose_id
                WHERE pd.session_id = ?
                ''', conn, params=(session_id,))
                
                # Process DataFrames
                session_data = session_df.to_dict('records')[0]
                
                # Parse landmark data
                pose_df['landmark_data'] = pose_df['landmark_data'].apply(json.loads)
                
                # Parse variability data
                metrics_df['variability'] = metrics_df['variability'].apply(json.loads)
                
                return {
                    'session': session_data,
                    'poses': pose_df.to_dict('records'),
                    'metrics': metrics_df.to_dict('records'),
                    'patterns': patterns_df.to_dict('records')
                }
                
        except Exception as e:
            logger.error(f"Error getting session data: {str(e)}")
            raise
            
    def get_movement_analysis(self, session_id: int) -> Dict:
        """Get movement analysis summary for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get metrics data
                metrics_df = pd.read_sql_query('''
                SELECT mm.* FROM movement_metrics mm
                JOIN pose_data pd ON mm.pose_id = pd.pose_id
                WHERE pd.session_id = ?
                ''', conn, params=(session_id,))
                
                # Get patterns data
                patterns_df = pd.read_sql_query('''
                SELECT mp.* FROM movement_patterns mp
                JOIN pose_data pd ON mp.pose_id = pd.pose_id
                WHERE pd.session_id = ?
                ''', conn, params=(session_id,))
                
                # Calculate summary statistics
                metrics_summary = {
                    'smoothness': {
                        'mean': float(metrics_df['smoothness'].mean()),
                        'std': float(metrics_df['smoothness'].std()),
                        'min': float(metrics_df['smoothness'].min()),
                        'max': float(metrics_df['smoothness'].max())
                    },
                    'complexity': {
                        'mean': float(metrics_df['complexity'].mean()),
                        'std': float(metrics_df['complexity'].std()),
                        'min': float(metrics_df['complexity'].min()),
                        'max': float(metrics_df['complexity'].max())
                    },
                    'periodicity': {
                        'mean': float(metrics_df['periodicity'].mean()),
                        'std': float(metrics_df['periodicity'].std()),
                        'min': float(metrics_df['periodicity'].min()),
                        'max': float(metrics_df['periodicity'].max())
                    }
                }
                
                # Analyze movement patterns
                pattern_summary = patterns_df.groupby('pattern_name').agg({
                    'confidence': ['mean', 'std'],
                    'frequency': ['mean', 'std'],
                    'duration': ['mean', 'std'],
                    'intensity': ['mean', 'std']
                }).to_dict()
                
                return {
                    'metrics_summary': metrics_summary,
                    'pattern_summary': pattern_summary
                }
                
        except Exception as e:
            logger.error(f"Error getting movement analysis: {str(e)}")
            raise
            
    def export_session_data(self, session_id: int, export_path: str):
        """Export session data to CSV files"""
        try:
            session_data = self.get_session_data(session_id)
            
            # Export to multiple CSV files
            pd.DataFrame([session_data['session']]).to_csv(
                f"{export_path}/session_{session_id}.csv", index=False
            )
            
            pd.DataFrame(session_data['poses']).to_csv(
                f"{export_path}/poses_{session_id}.csv", index=False
            )
            
            pd.DataFrame(session_data['metrics']).to_csv(
                f"{export_path}/metrics_{session_id}.csv", index=False
            )
            
            pd.DataFrame(session_data['patterns']).to_csv(
                f"{export_path}/patterns_{session_id}.csv", index=False
            )
            
            logger.info(f"Session data exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting session data: {str(e)}")
            raise
