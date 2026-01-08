"""
BaseStage - Foundation for All Processing Stages
================================================

Minimal, effective base class that all 11 stages inherit from.
Handles common operations so stages can focus on business logic.

Features:
- Automatic logging
- Input validation
- Timing
- Error handling
- Consistent output format
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import time

from utils import get_logger, validate_dataframe, format_output_dict
from utils.constants import STAGE_NAMES


class BaseStage(ABC):
    """Base class for all processing stages."""
    
    def __init__(self, stage_number: int):
        """
        Initialize stage.
        
        Args:
            stage_number: Stage number (1-11)
        """
        self.stage_number = stage_number
        self.stage_name = STAGE_NAMES.get(stage_number, f"Stage {stage_number}")
        self.logger = get_logger(f"Stage{stage_number}")
        
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute stage with logging and error handling.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results, metadata, and status
        """
        start_time = time.time()
        
        try:
            # Log start
            self.logger.info("="*60)
            self.logger.info(f"🚀 STARTING: {self.stage_name}")
            self.logger.info(f"   Input rows: {len(data)}")
            
            # Validate input
            self._validate_input(data)
            
            # Execute business logic (implemented by child class)
            result = self.process(data, **kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Build output
            output = {
                'stage_number': self.stage_number,
                'stage_name': self.stage_name,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': round(duration, 2),
                'input_records': len(data),
                'results': result
            }
            
            # Log completion
            self.logger.info(f"✅ COMPLETED: {self.stage_name}")
            self.logger.info(f"   Duration: {duration:.2f}s")
            self.logger.info("="*60)
            
            return output
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"❌ FAILED: {self.stage_name}", exc_info=True)
            
            return {
                'stage_number': self.stage_number,
                'stage_name': self.stage_name,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': round(duration, 2)
            }
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Process data - implemented by each stage.
        
        Args:
            data: Input DataFrame
            **kwargs: Stage-specific parameters
            
        Returns:
            Dictionary with stage-specific results
        """
        pass
    
    def _validate_input(self, data: pd.DataFrame):
        """
        Validate input data.
        
        Args:
            data: Input DataFrame
        """
        validate_dataframe(data, min_rows=1, df_name=self.stage_name)
        
        # Child classes can override for specific column checks
        required = self._get_required_columns()
        if required:
            validate_dataframe(data, required_columns=required, df_name=self.stage_name)
    
    def _get_required_columns(self) -> Optional[list]:
        """
        Get required columns for this stage.
        Override in child classes.
        
        Returns:
            List of required column names or None
        """
        return None
