import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import json
from datetime import datetime
import logging
from pathlib import Path
from uuid import uuid4
from ..types import MovementPattern, ComponentHealth

logger = logging.getLogger(__name__)

class UnifiedDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Create subjects table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS subjects (
                        subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        age INTEGER,
                        gender TEXT,
                        height REAL,
                        weight REAL,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject_id INTEGER,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        notes TEXT,
                        FOREIGN KEY (subject_id) REFERENCES subjects (subject_id)
                    )
                ''')
                
                # Create metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metric_type TEXT,
                        metric_value REAL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
            
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(str(self.db_path))
        
    def add_subject(self, subject_data: Dict[str, Any]) -> int:
        """Add new subject to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                now = datetime.now().isoformat()
                subject_data.update({
                    'created_at': now,
                    'updated_at': now
                })
                
                columns = ', '.join(subject_data.keys())
                placeholders = ', '.join(['?' for _ in subject_data])
                query = f'''
                    INSERT INTO subjects ({columns})
                    VALUES ({placeholders})
                '''
                
                cursor.execute(query, list(subject_data.values()))
                conn.commit()
                
                return cursor.lastrowid or 0  # Return 0 if no id generated
                
        except Exception as e:
            logger.error(f"Error adding subject: {str(e)}")
            raise
            
    def start_session(self, subject_id: str, session_type: str, 
                     environment_data: Optional[Dict[str, Any]] = None) -> int:
        """Start new recording session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO sessions (
                    subject_id, start_time, session_type, 
                    environment_data, notes
                )
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    subject_id,
                    datetime.now().isoformat(),
                    session_type,
                    json.dumps(environment_data or {}),
                    ""
                ))
                
                session_id = cursor.lastrowid
                if session_id is None:
                    raise ValueError("Failed to get session ID")
                conn.commit()
                return session_id
                
        except Exception as e:
            logger.error(f"Error starting session: {str(e)}")
            raise
            
    def store_facial_data(self, session_id: Union[str, int], data: Dict[str, Any]) -> None:
        """Store facial analysis data"""
        session_id = int(session_id) if isinstance(session_id, str) else session_id
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO facial_data (
                        session_id, timestamp, landmarks, expressions,
                        emotions, pupil_metrics, micro_expressions
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    datetime.now().isoformat(),
                    json.dumps(data.get('landmarks', {})),
                    json.dumps(data.get('expressions', {})),
                    json.dumps(data.get('emotions', {})),
                    json.dumps(data.get('pupil_metrics', {})),
                    json.dumps(data.get('micro_expressions', {}))
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing facial data: {e}")
            raise
            
    def store_body_data(self, session_id: str, data: Dict[str, Any]) -> None:
        """Store body analysis data"""
        try:
            if not isinstance(session_id, str):
                session_id = str(session_id)
            
            # Convert numpy arrays and custom objects to JSON-serializable format
            processed_data = self._prepare_data_for_storage(data)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO body_metrics 
                    (session_id, timestamp, metrics_data)
                    VALUES (?, ?, ?)
                """, (
                    session_id,
                    datetime.now().isoformat(),
                    json.dumps(processed_data)
                ))
                
        except Exception as e:
            logger.error(f"Error storing body data: {e}")
            raise
            
    def _prepare_data_for_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for storage"""
        def convert_value(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (list, tuple)):
                return [convert_value(x) for x in v]
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            if hasattr(v, 'dtype'):
                if np.issubdtype(v.dtype, np.integer):
                    return int(v)
                if np.issubdtype(v.dtype, np.floating):
                    return float(v)
            if hasattr(v, 'to_dict'):
                return v.to_dict()
            return v
        
        return {k: convert_value(v) for k, v in data.items()}
            
    def store_analysis_results(self, session_id: int, analysis_type: str,
                             results: Dict):
        """Store analysis results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO analysis_results (
                    session_id, timestamp, analysis_type,
                    metrics, patterns, predictions, correlations
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    datetime.now().isoformat(),
                    analysis_type,
                    json.dumps(results.get('metrics', {})),
                    json.dumps(results.get('patterns', {})),
                    json.dumps(results.get('predictions', {})),
                    json.dumps(results.get('correlations', {}))
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing analysis results: {str(e)}")
            raise
            
    def update_historical_patterns(self, subject_id: str, 
                                 patterns: List[Dict]):
        """Update historical patterns for subject"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                for pattern in patterns:
                    # Check if pattern exists
                    cursor.execute('''
                    SELECT pattern_id, occurrence_count 
                    FROM historical_patterns
                    WHERE subject_id = ? AND pattern_type = ?
                    ''', (subject_id, pattern['type']))
                    
                    result = cursor.fetchone()
                    
                    if result:
                        # Update existing pattern
                        pattern_id, count = result
                        cursor.execute('''
                        UPDATE historical_patterns
                        SET pattern_data = ?,
                            confidence = ?,
                            last_observed = ?,
                            occurrence_count = ?
                        WHERE pattern_id = ?
                        ''', (
                            json.dumps(pattern['data']),
                            pattern['confidence'],
                            now,
                            count + 1,
                            pattern_id
                        ))
                    else:
                        # Insert new pattern
                        cursor.execute('''
                        INSERT INTO historical_patterns (
                            subject_id, pattern_type, pattern_data,
                            confidence, first_observed, last_observed,
                            occurrence_count
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            subject_id,
                            pattern['type'],
                            json.dumps(pattern['data']),
                            pattern['confidence'],
                            now,
                            now,
                            1
                        ))
                        
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating historical patterns: {str(e)}")
            raise
            
    def calculate_correlations(self, subject_id: str):
        """Calculate correlations between different metrics"""
        try:
            with self.get_connection() as conn:
                # Get all metrics for subject
                facial_df = pd.read_sql_query('''
                SELECT f.* FROM facial_data f
                JOIN sessions s ON f.session_id = s.session_id
                WHERE s.subject_id = ?
                ''', conn, params=(subject_id,))
                
                body_df = pd.read_sql_query('''
                SELECT b.* FROM body_data b
                JOIN sessions s ON b.session_id = s.session_id
                WHERE s.subject_id = ?
                ''', conn, params=(subject_id,))
                
                # Extract metrics from JSON
                metrics = {}
                
                for col in facial_df.columns:
                    if col.endswith('metrics'):
                        metrics[f'facial_{col}'] = facial_df[col].apply(json.loads)
                        
                for col in body_df.columns:
                    if col.endswith('metrics'):
                        metrics[f'body_{col}'] = body_df[col].apply(json.loads)
                        
                # Calculate correlations
                correlations = []
                now = datetime.now().isoformat()
                
                for metric1 in metrics:
                    for metric2 in metrics:
                        if metric1 >= metric2:
                            continue
                            
                        corr = metrics[metric1].corr(metrics[metric2])
                        if not np.isnan(corr):
                            correlations.append({
                                'subject_id': subject_id,
                                'metric_1': metric1,
                                'metric_2': metric2,
                                'correlation_value': float(corr),
                                'p_value': 0.0,  # Calculate proper p-value
                                'timestamp': now
                            })
                            
                # Store correlations
                cursor = conn.cursor()
                for corr in correlations:
                    cursor.execute('''
                    INSERT INTO correlations (
                        subject_id, metric_1, metric_2,
                        correlation_value, p_value, timestamp
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        corr['subject_id'],
                        corr['metric_1'],
                        corr['metric_2'],
                        corr['correlation_value'],
                        corr['p_value'],
                        corr['timestamp']
                    ))
                    
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            raise
            
    def get_subject_analysis(self, subject_id: str) -> Dict:
        """Get comprehensive analysis for subject"""
        try:
            with self.get_connection() as conn:
                # Get subject info
                subject_df = pd.read_sql_query('''
                SELECT * FROM subjects WHERE subject_id = ?
                ''', conn, params=(subject_id,))
                
                # Get historical patterns
                patterns_df = pd.read_sql_query('''
                SELECT * FROM historical_patterns
                WHERE subject_id = ?
                ORDER BY last_observed DESC
                ''', conn, params=(subject_id,))
                
                # Get recent correlations
                correlations_df = pd.read_sql_query('''
                SELECT * FROM correlations
                WHERE subject_id = ?
                ORDER BY timestamp DESC
                LIMIT 100
                ''', conn, params=(subject_id,))
                
                # Get recent analysis results
                analysis_df = pd.read_sql_query('''
                SELECT ar.* FROM analysis_results ar
                JOIN sessions s ON ar.session_id = s.session_id
                WHERE s.subject_id = ?
                ORDER BY ar.timestamp DESC
                LIMIT 100
                ''', conn, params=(subject_id,))
                
                return {
                    'subject_info': subject_df.to_dict('records')[0],
                    'historical_patterns': patterns_df.to_dict('records'),
                    'correlations': correlations_df.to_dict('records'),
                    'recent_analysis': analysis_df.to_dict('records')
                }
                
        except Exception as e:
            logger.error(f"Error getting subject analysis: {str(e)}")
            raise
            
    def export_subject_data(self, subject_id: str, export_path: str) -> None:
        """Export all data for a subject"""
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            with self.get_connection() as conn:
                tables = [
                    'subjects', 'sessions', 'facial_data', 'body_data',
                    'analysis_results', 'historical_patterns', 'correlations'
                ]
                
                for table in tables:
                    df = pd.read_sql_query(f'''
                    SELECT * FROM {table}
                    WHERE subject_id = ?
                    ''', conn, params=(subject_id,))
                    
                    if not df.empty:
                        output_file = export_dir / f'{table}_{subject_id}.csv'
                        df.to_csv(str(output_file), index=False)
                        
            logger.info(f"Subject data exported to {export_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting subject data: {str(e)}")
            raise
            
    def create_session(self, subject_id: str, session_type: str) -> str:
        """Crea una nuova sessione"""
        session_id = str(uuid4())
        timestamp = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, subject_id, type, start_time)
                VALUES (?, ?, ?, ?)
            """, (session_id, subject_id, session_type, timestamp))
            
        return session_id
        
    def finish_session(self, session_id: str) -> None:
        """Termina una sessione"""
        timestamp = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET end_time = ?, status = 'completed'
                WHERE session_id = ?
            """, (timestamp, session_id))
