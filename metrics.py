# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Metrics for Precipitation Prediction
=============================================================
This module contains all evaluation metrics and visualization tools
for precipitation prediction models.

Usage:
    from evaluation_metrics import PrecipitationMetrics, MetricsVisualizer, EvaluationPipeline
    
    # Initialize metrics calculator
    metrics_calc = PrecipitationMetrics(threshold=0.1)
    
    # Calculate all metrics
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred)
    
    # Print results
    metrics_calc.print_metrics(metrics)
    
    # Save results
    metrics_calc.save_metrics(metrics, 'path/to/save/metrics.json')
"""

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
from scipy.ndimage import uniform_filter
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PRECIPITATION METRICS CLASS
# ============================================================
class PrecipitationMetrics:
    """
    Comprehensive evaluation metrics for precipitation prediction models.
    
    This class provides methods to calculate various metrics for evaluating
    precipitation prediction models, including:
    - Classification metrics (Precision, Recall, F1, Accuracy)
    - Precipitation-specific metrics (FAR, POD, CSI, HSS, FSS, ETS, GSS)
    - Regression metrics (MSE, RMSE, MAE, MAPE)
    - Efficiency metrics (KGE, NSE)
    
    Parameters:
    -----------
    threshold : float, default=0.1
        Threshold for converting continuous predictions to binary 
        (precipitation/no-precipitation)
    epsilon : float, default=1e-7
        Small value to avoid division by zero
    """
    
    def __init__(self, threshold: float = 0.1, epsilon: float = 1e-7):
        self.threshold = threshold
        self.epsilon = epsilon
        
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert input to numpy array"""
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.array(data)
    
    def _binarize(self, data: np.ndarray) -> np.ndarray:
        """Convert continuous values to binary using threshold"""
        return (data > self.threshold).astype(np.float32)
    
    # ============================================================
    # CONFUSION MATRIX COMPONENTS
    # ============================================================
    def confusion_matrix_components(self, 
                                   y_true: Union[np.ndarray, torch.Tensor], 
                                   y_pred: Union[np.ndarray, torch.Tensor],
                                   binarize: bool = True) -> Dict[str, int]:
        """
        Calculate confusion matrix components: TP, TN, FP, FN
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        y_pred : array-like
            Predicted values
        binarize : bool, default=True
            Whether to binarize the values using threshold
            
        Returns:
        --------
        dict : Dictionary containing TP, TN, FP, FN counts and total
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        if binarize:
            y_true_bin = self._binarize(y_true)
            y_pred_bin = self._binarize(y_pred)
        else:
            y_true_bin = y_true
            y_pred_bin = y_pred
        
        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
        
        return {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'total': int(tp + tn + fp + fn)
        }
    
    # ============================================================
    # BASIC CLASSIFICATION METRICS
    # ============================================================
    def precision(self, 
                  y_true: Union[np.ndarray, torch.Tensor], 
                  y_pred: Union[np.ndarray, torch.Tensor],
                  binarize: bool = True) -> float:
        """
        Calculate Precision (Positive Predictive Value)
        
        Precision = TP / (TP + FP)
        
        Measures the proportion of correctly predicted precipitation events
        among all predicted precipitation events.
        
        Returns:
        --------
        float : Precision score [0, 1]
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        if binarize:
            y_true_bin = self._binarize(y_true)
            y_pred_bin = self._binarize(y_pred)
        else:
            y_true_bin = y_true
            y_pred_bin = y_pred
            
        return precision_score(y_true_bin, y_pred_bin, zero_division=0)
    
    def recall(self, 
              y_true: Union[np.ndarray, torch.Tensor], 
              y_pred: Union[np.ndarray, torch.Tensor],
              binarize: bool = True) -> float:
        """
        Calculate Recall (Sensitivity, True Positive Rate, Hit Rate)
        
        Recall = TP / (TP + FN)
        
        Measures the proportion of actual precipitation events that were
        correctly predicted.
        
        Returns:
        --------
        float : Recall score [0, 1]
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        if binarize:
            y_true_bin = self._binarize(y_true)
            y_pred_bin = self._binarize(y_pred)
        else:
            y_true_bin = y_true
            y_pred_bin = y_pred
            
        return recall_score(y_true_bin, y_pred_bin, zero_division=0)
    
    def f1_score(self, 
                y_true: Union[np.ndarray, torch.Tensor], 
                y_pred: Union[np.ndarray, torch.Tensor],
                binarize: bool = True) -> float:
        """
        Calculate F1 Score (Harmonic mean of Precision and Recall)
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Provides a balanced measure between precision and recall.
        
        Returns:
        --------
        float : F1 score [0, 1]
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        if binarize:
            y_true_bin = self._binarize(y_true)
            y_pred_bin = self._binarize(y_pred)
        else:
            y_true_bin = y_true
            y_pred_bin = y_pred
            
        return f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    # ============================================================
    # FALSE ALARM RATE (FAR)
    # ============================================================
    def false_alarm_rate(self, 
                         y_true: Union[np.ndarray, torch.Tensor], 
                         y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate False Alarm Rate (FAR)
        
        FAR = FP / (TP + FP)
        
        Measures the proportion of predicted precipitation events that
        were false alarms. Lower values indicate better performance.
        
        Returns:
        --------
        float : False Alarm Rate [0, 1], lower is better
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        cm = self.confusion_matrix_components(y_true, y_pred)
        
        tp = cm['TP']
        fp = cm['FP']
        
        if (tp + fp) == 0:
            return 1.0  # Worst case
        
        return fp / (tp + fp + self.epsilon)
    
    # ============================================================
    # PROBABILITY OF DETECTION (POD)
    # ============================================================
    def probability_of_detection(self, 
                                 y_true: Union[np.ndarray, torch.Tensor], 
                                 y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Probability of Detection (POD) / Hit Rate
        
        POD = TP / (TP + FN)
        
        Measures the fraction of actual precipitation events that were
        correctly predicted. Higher values indicate better detection.
        
        Returns:
        --------
        float : POD [0, 1], higher is better
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        cm = self.confusion_matrix_components(y_true, y_pred)
        
        tp = cm['TP']
        fn = cm['FN']
        
        if (tp + fn) == 0:
            return 0.0  # Worst case
        
        return tp / (tp + fn + self.epsilon)
    
    def hit_rate(self, 
                y_true: Union[np.ndarray, torch.Tensor], 
                y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """Alias for probability_of_detection"""
        return self.probability_of_detection(y_true, y_pred)
    
    # ============================================================
    # CRITICAL SUCCESS INDEX (CSI)
    # ============================================================
    def critical_success_index(self, 
                               y_true: Union[np.ndarray, torch.Tensor], 
                               y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Critical Success Index (CSI) / Threat Score
        
        CSI = TP / (TP + FP + FN)
        
        Measures the fraction of correctly predicted precipitation events,
        accounting for both false alarms and missed events.
        
        Returns:
        --------
        float : CSI [0, 1], higher is better
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        cm = self.confusion_matrix_components(y_true, y_pred)
        
        tp = cm['TP']
        fp = cm['FP']
        fn = cm['FN']
        
        denominator = tp + fp + fn
        if denominator == 0:
            return 0.0
        
        return tp / (denominator + self.epsilon)
    
    # ============================================================
    # HEIDKE SKILL SCORE (HSS)
    # ============================================================
    def heidke_skill_score(self, 
                           y_true: Union[np.ndarray, torch.Tensor], 
                           y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Heidke Skill Score (HSS)
        
        HSS = 2 * (TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
        
        Measures the fraction of correct predictions after eliminating
        those that would be correct by random chance.
        
        Returns:
        --------
        float : HSS [-1, 1], 1 is perfect, 0 is no skill
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        cm = self.confusion_matrix_components(y_true, y_pred)
        
        tp = cm['TP']
        tn = cm['TN']
        fp = cm['FP']
        fn = cm['FN']
        
        numerator = 2 * (tp * tn - fp * fn)
        denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        
        if denominator == 0:
            return 0.0
        
        return numerator / (denominator + self.epsilon)
    
    # ============================================================
    # GILBERT SKILL SCORE (GSS) / EQUITABLE THREAT SCORE (ETS)
    # ============================================================
    def gilbert_skill_score(self, 
                           y_true: Union[np.ndarray, torch.Tensor], 
                           y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Gilbert Skill Score (GSS) / Equitable Threat Score (ETS)
        
        GSS = (TP - random_hits) / (TP + FP + FN - random_hits)
        where random_hits = (TP + FP) * (TP + FN) / total
        
        Adjusts CSI for hits that could occur by random chance.
        Also known as the Equitable Threat Score (ETS).
        
        Returns:
        --------
        float : GSS [-1/3, 1], higher is better, 0 indicates no skill
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        cm = self.confusion_matrix_components(y_true, y_pred)
        
        tp = cm['TP']
        fp = cm['FP']
        fn = cm['FN']
        total = cm['total']
        
        if total == 0:
            return 0.0
        
        # Random hits (expected correct forecasts by chance)
        random_hits = ((tp + fp) * (tp + fn)) / total
        
        denominator = tp + fp + fn - random_hits
        if denominator == 0:
            return 0.0
        
        return (tp - random_hits) / (denominator + self.epsilon)
    
    # Alias for backward compatibility
    def equitable_threat_score(self, 
                              y_true: Union[np.ndarray, torch.Tensor], 
                              y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """Alias for gilbert_skill_score"""
        return self.gilbert_skill_score(y_true, y_pred)
    
    # ============================================================
    # FRACTIONS SKILL SCORE (FSS)
    # ============================================================
    def fractions_skill_score(self, 
                             y_true: Union[np.ndarray, torch.Tensor], 
                             y_pred: Union[np.ndarray, torch.Tensor],
                             window_size: int = 5) -> float:
        """
        Calculate Fractions Skill Score (FSS)
        
        FSS measures spatial skill by comparing fractions of precipitation
        in neighborhoods of different sizes.
        
        Parameters:
        -----------
        window_size : int, default=5
            Size of the neighborhood window for computing fractions
            
        Returns:
        --------
        float : FSS [0, 1], 1 is perfect
        """
        y_true = self._to_numpy(y_true).squeeze()
        y_pred = self._to_numpy(y_pred).squeeze()
        
        # Binarize
        y_true_bin = self._binarize(y_true)
        y_pred_bin = self._binarize(y_pred)
        
        if len(y_true_bin.shape) == 1:
            # For 1D data, use correlation
            if np.std(y_true_bin) > 0 and np.std(y_pred_bin) > 0:
                return np.corrcoef(y_true_bin, y_pred_bin)[0, 1]
            return 0.0
        
        # For 2D data, use moving window
        obs_frac = uniform_filter(y_true_bin.astype(float), size=window_size)
        fcst_frac = uniform_filter(y_pred_bin.astype(float), size=window_size)
        
        # Calculate MSE of fractions
        mse_f = np.mean((obs_frac - fcst_frac) ** 2)
        
        # Calculate reference MSE
        mse_ref = np.mean(obs_frac ** 2) + np.mean(fcst_frac ** 2)
        
        if mse_ref == 0:
            return 1.0
        
        return 1.0 - mse_f / (mse_ref + self.epsilon)
    
    # ============================================================
    # KLING-GUPTA EFFICIENCY (KGE)
    # ============================================================
    def kling_gupta_efficiency(self, 
                               y_true: Union[np.ndarray, torch.Tensor], 
                               y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Kling-Gupta Efficiency (KGE)
        
        KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
        where:
            r = Pearson correlation coefficient
            α = σ_pred / σ_true (variability ratio)
            β = μ_pred / μ_true (bias ratio)
        
        Combines correlation, bias, and variability into a single metric.
        
        Returns:
        --------
        float : KGE [-inf, 1], 1 is perfect
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) < 2:
            return -np.inf
        
        # Pearson correlation
        r = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Standard deviation ratio (alpha)
        std_true = np.std(y_true)
        std_pred = np.std(y_pred)
        
        if std_true == 0:
            alpha = 1.0 if std_pred == 0 else 0.0
        else:
            alpha = std_pred / (std_true + self.epsilon)
        
        # Mean ratio (beta) - bias
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        
        if mean_true == 0:
            beta = 1.0 if mean_pred == 0 else 0.0
        else:
            beta = mean_pred / (mean_true + self.epsilon)
        
        # KGE calculation
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        return float(kge)
    
    # ============================================================
    # REGRESSION METRICS
    # ============================================================
    def mean_squared_error(self, 
                          y_true: Union[np.ndarray, torch.Tensor], 
                          y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Mean Squared Error (MSE)
        
        MSE = (1/n) * Σ(y_true - y_pred)²
        
        Returns:
        --------
        float : MSE
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        return float(np.mean((y_true - y_pred) ** 2))
    
    def root_mean_squared_error(self, 
                               y_true: Union[np.ndarray, torch.Tensor], 
                               y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Root Mean Squared Error (RMSE)
        
        RMSE = sqrt(MSE)
        
        Returns:
        --------
        float : RMSE
        """
        return float(np.sqrt(self.mean_squared_error(y_true, y_pred)))
    
    def mean_absolute_error(self, 
                           y_true: Union[np.ndarray, torch.Tensor], 
                           y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Mean Absolute Error (MAE)
        
        MAE = (1/n) * Σ|y_true - y_pred|
        
        Returns:
        --------
        float : MAE
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def mean_absolute_percentage_error(self, 
                                      y_true: Union[np.ndarray, torch.Tensor], 
                                      y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE)
        
        MAPE = (100/n) * Σ|(y_true - y_pred) / y_true|
        
        Returns:
        --------
        float : MAPE in percentage
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        # Avoid division by zero
        mask = y_true != 0
        if np.sum(mask) == 0:
            return np.inf
        
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        
        percentage_errors = np.abs((y_true_masked - y_pred_masked) / (y_true_masked + self.epsilon))
        return float(np.mean(percentage_errors) * 100)
    
    # ============================================================
    # ADDITIONAL METRICS
    # ============================================================
    def bias_score(self, 
                  y_true: Union[np.ndarray, torch.Tensor], 
                  y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Frequency Bias Score
        
        Bias = (TP + FP) / (TP + FN)
        
        Indicates whether the model tends to under-forecast (<1) or 
        over-forecast (>1) precipitation events.
        
        Returns:
        --------
        float : Bias score, 1 is unbiased
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        cm = self.confusion_matrix_components(y_true, y_pred)
        
        tp = cm['TP']
        fp = cm['FP']
        fn = cm['FN']
        
        if (tp + fn) == 0:
            return 0.0
        
        return (tp + fp) / (tp + fn + self.epsilon)
    
    def accuracy(self, 
                y_true: Union[np.ndarray, torch.Tensor], 
                y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Overall Accuracy
        
        Accuracy = (TP + TN) / Total
        
        Returns:
        --------
        float : Accuracy [0, 1]
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        cm = self.confusion_matrix_components(y_true, y_pred)
        
        return (cm['TP'] + cm['TN']) / (cm['total'] + self.epsilon)
    
    def nash_sutcliffe_efficiency(self, 
                                 y_true: Union[np.ndarray, torch.Tensor], 
                                 y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Nash-Sutcliffe Efficiency (NSE)
        
        NSE = 1 - [Σ(y_true - y_pred)² / Σ(y_true - μ_true)²]
        
        Commonly used in hydrology to assess predictive skill.
        
        Returns:
        --------
        float : NSE [-inf, 1], 1 is perfect
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if denominator == 0:
            return -np.inf
        
        return float(1 - numerator / (denominator + self.epsilon))
    
    def correlation_coefficient(self, 
                               y_true: Union[np.ndarray, torch.Tensor], 
                               y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate Pearson Correlation Coefficient
        
        Returns:
        --------
        float : Correlation coefficient [-1, 1]
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) < 2:
            return 0.0
        
        return float(np.corrcoef(y_true, y_pred)[0, 1])
    
    def coefficient_of_determination(self, 
                                    y_true: Union[np.ndarray, torch.Tensor], 
                                    y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate R² (Coefficient of Determination)
        
        Returns:
        --------
        float : R² [-inf, 1], 1 is perfect
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return float(1 - ss_res / (ss_tot + self.epsilon))
    
    # ============================================================
    # COMPREHENSIVE METRICS CALCULATION
    # ============================================================
    def calculate_all_metrics(self, 
                             y_true: Union[np.ndarray, torch.Tensor], 
                             y_pred: Union[np.ndarray, torch.Tensor],
                             window_size: int = 5,
                             thresholds: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        y_pred : array-like
            Predicted values
        window_size : int, default=5
            Window size for FSS calculation
        thresholds : list of float, optional
            Additional thresholds for threshold-dependent metrics
            
        Returns:
        --------
        dict : Dictionary containing all calculated metrics
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        metrics = {}
        
        # Regression metrics
        metrics['MSE'] = self.mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = self.root_mean_squared_error(y_true, y_pred)
        metrics['MAE'] = self.mean_absolute_error(y_true, y_pred)
        metrics['MAPE'] = self.mean_absolute_percentage_error(y_true, y_pred)
        
        # Correlation metrics
        metrics['Correlation'] = self.correlation_coefficient(y_true, y_pred)
        metrics['R2'] = self.coefficient_of_determination(y_true, y_pred)
        
        # Efficiency metrics
        metrics['KGE'] = self.kling_gupta_efficiency(y_true, y_pred)
        metrics['NSE'] = self.nash_sutcliffe_efficiency(y_true, y_pred)
        
        # Classification metrics (using default threshold)
        metrics['Precision'] = self.precision(y_true, y_pred)
        metrics['Recall'] = self.recall(y_true, y_pred)
        metrics['F1_Score'] = self.f1_score(y_true, y_pred)
        metrics['Accuracy'] = self.accuracy(y_true, y_pred)
        
        # Precipitation-specific metrics
        metrics['FAR'] = self.false_alarm_rate(y_true, y_pred)
        metrics['POD'] = self.probability_of_detection(y_true, y_pred)
        metrics['CSI'] = self.critical_success_index(y_true, y_pred)
        metrics['HSS'] = self.heidke_skill_score(y_true, y_pred)
        metrics['FSS'] = self.fractions_skill_score(y_true, y_pred, window_size)
        metrics['GSS'] = self.gilbert_skill_score(y_true, y_pred)  # Gilbert Skill Score
        metrics['ETS'] = self.equitable_threat_score(y_true, y_pred)  # Alias for GSS
        metrics['Bias'] = self.bias_score(y_true, y_pred)
        
        # Calculate metrics for different thresholds if provided
        if thresholds is not None:
            for thresh in thresholds:
                original_threshold = self.threshold
                self.threshold = thresh
                
                metrics[f'POD_thresh_{thresh}'] = self.probability_of_detection(y_true, y_pred)
                metrics[f'FAR_thresh_{thresh}'] = self.false_alarm_rate(y_true, y_pred)
                metrics[f'CSI_thresh_{thresh}'] = self.critical_success_index(y_true, y_pred)
                metrics[f'HSS_thresh_{thresh}'] = self.heidke_skill_score(y_true, y_pred)
                metrics[f'GSS_thresh_{thresh}'] = self.gilbert_skill_score(y_true, y_pred)
                
                self.threshold = original_threshold
        
        # Confusion matrix components
        cm = self.confusion_matrix_components(y_true, y_pred)
        metrics.update({f'CM_{k}': v for k, v in cm.items()})
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Pretty print all metrics
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metrics to print
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION METRICS")
        print("="*70)
        
        print("\n REGRESSION METRICS:")
        print("-" * 40)
        regression_metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'Correlation', 'R2']
        for metric in regression_metrics:
            if metric in metrics:
                value = metrics[metric]
                if metric == 'MAPE':
                    print(f"  {metric:<15}: {value:.2f}%")
                else:
                    print(f"  {metric:<15}: {value:.6f}")
        
        print("\n CLASSIFICATION METRICS:")
        print("-" * 40)
        classification_metrics = ['Precision', 'Recall', 'F1_Score', 'Accuracy']
        for metric in classification_metrics:
            if metric in metrics:
                print(f"  {metric:<15}: {metrics[metric]:.4f}")
        
        print("\n PRECIPITATION-SPECIFIC METRICS:")
        print("-" * 40)
        precip_metrics = ['FAR', 'POD', 'CSI', 'HSS', 'GSS', 'ETS', 'FSS', 'Bias']
        for metric in precip_metrics:
            if metric in metrics:
                print(f"  {metric:<15}: {metrics[metric]:.4f}")
        
        print("\n EFFICIENCY METRICS:")
        print("-" * 40)
        efficiency_metrics = ['KGE', 'NSE']
        for metric in efficiency_metrics:
            if metric in metrics:
                print(f"  {metric:<15}: {metrics[metric]:.4f}")
        
        print("\n CONFUSION MATRIX:")
        print("-" * 40)
        cm_metrics = ['CM_TP', 'CM_TN', 'CM_FP', 'CM_FN', 'CM_total']
        for metric in cm_metrics:
            if metric in metrics:
                print(f"  {metric:<15}: {metrics[metric]}")
        
        print("="*70 + "\n")
    
    def save_metrics(self, 
                    metrics: Dict[str, float], 
                    save_path: Union[str, Path]):
        """
        Save metrics to JSON file
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metrics to save
        save_path : str or Path
            Path to save the JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        print(f" Metrics saved to: {save_path}")
    
    def get_metrics_dataframe(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """
        Convert metrics dictionary to pandas DataFrame
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metrics
            
        Returns:
        --------
        pd.DataFrame : DataFrame with metrics
        """
        # Filter out confusion matrix components for cleaner display
        filtered_metrics = {k: v for k, v in metrics.items() 
                          if not k.startswith('CM_')}
        
        df = pd.DataFrame([filtered_metrics])
        return df.T.rename(columns={0: 'Value'})


# ============================================================
# METRICS VISUALIZER CLASS
# ============================================================
class MetricsVisualizer:
    """Visualization tools for evaluation metrics"""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             threshold: float = 0.1, 
                             save_path: Optional[Path] = None,
                             title: str = "Confusion Matrix"):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        y_pred : array-like
            Predicted values
        threshold : float
            Threshold for binarization
        save_path : Path, optional
            Path to save the figure
        title : str
            Plot title
        """
        y_true_bin = (y_true.flatten() > threshold).astype(int)
        y_pred_bin = (y_pred.flatten() > threshold).astype(int)
        
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=['No Rain', 'Rain'],
               yticklabels=['No Rain', 'Rain'],
               xlabel='Predicted', ylabel='Actual',
               title=f'{title} (threshold={threshold})')
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / cm.sum() * 100
                text = f'{cm[i, j]}\n({percentage:.1f}%)'
                ax.text(j, i, text,
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                               metric_names: List[str],
                               title: str = "Metrics Comparison",
                               save_path: Optional[Path] = None,
                               figsize: Tuple[int, int] = (12, 6)):
        """
        Plot comparison of metrics across different models
        
        Parameters:
        -----------
        metrics_dict : dict
            Dictionary mapping model names to their metrics dictionaries
        metric_names : list
            List of metric names to plot
        title : str
            Plot title
        save_path : Path, optional
            Path to save the figure
        figsize : tuple
            Figure size
        """
        n_metrics = len(metric_names)
        n_models = len(metrics_dict)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))
        
        for i, (model_name, metrics) in enumerate(metrics_dict.items()):
            values = [metrics.get(m, 0) for m in metric_names]
            bars = ax.bar(x + i * width, values, width, 
                         label=model_name, alpha=0.8, color=colors[i])
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val != 0:
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', 
                           fontsize=8, rotation=90)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_scatter_comparison(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               title: str = "Predicted vs Actual Precipitation",
                               save_path: Optional[Path] = None,
                               figsize: Tuple[int, int] = (8, 8)):
        """
        Create scatter plot of predicted vs actual values with density
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        y_pred : array-like
            Predicted values
        title : str
            Plot title
        save_path : Path, optional
            Path to save the figure
        figsize : tuple
            Figure size
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Remove zero-zero pairs for better visualization
        mask = (y_true > 0) | (y_pred > 0)
        y_true_plot = y_true[mask]
        y_pred_plot = y_pred[mask]
        
        # Sample if too many points
        if len(y_true_plot) > 10000:
            idx = np.random.choice(len(y_true_plot), 10000, replace=False)
            y_true_plot = y_true_plot[idx]
            y_pred_plot = y_pred_plot[idx]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Density scatter plot
        hist, xedges, yedges = np.histogram2d(y_true_plot, y_pred_plot, bins=50)
        im = ax.pcolormesh(xedges, yedges, hist.T, cmap='YlOrRd', alpha=0.8)
        plt.colorbar(im, ax=ax, label='Count')
        
        # Perfect prediction line
        max_val = max(y_true_plot.max(), y_pred_plot.max())
        ax.plot([0, max_val], [0, max_val], 'b--', linewidth=2, label='Perfect', alpha=0.7)
        
        # Regression line
        if len(y_true_plot) > 1:
            coeffs = np.polyfit(y_true_plot, y_pred_plot, 1)
            x_line = np.array([0, max_val])
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, 'r-', linewidth=2, 
                   label=f'Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}', alpha=0.7)
            
            # Add R² value
            correlation = np.corrcoef(y_true_plot, y_pred_plot)[0, 1]
            r2 = correlation ** 2
            ax.text(0.05, 0.95, f'R² = {r2:.4f}\nCorr = {correlation:.4f}',
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Actual Precipitation (mm)', fontsize=12)
        ax.set_ylabel('Predicted Precipitation (mm)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      save_path: Optional[Path] = None):
        """
        Plot ROC curve for binary precipitation detection
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        y_pred : array-like
            Predicted probabilities/values
        save_path : Path, optional
            Path to save the figure
        """
        from sklearn.metrics import roc_curve, auc
        
        y_true_bin = (y_true.flatten() > 0.1).astype(int)
        y_pred_flat = y_pred.flatten()
        
        fpr, tpr, thresholds = roc_curve(y_true_bin, y_pred_flat)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.3)
        
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title('ROC Curve for Precipitation Detection', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ============================================================
# EVALUATION PIPELINE CLASS
# ============================================================
class EvaluationPipeline:
    """
    Complete evaluation pipeline for precipitation prediction models.
    
    This class integrates model evaluation with metrics calculation,
    visualization, and result saving.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained PyTorch model
    device : str, default='cpu'
        Device to run evaluation on ('cpu' or 'cuda')
    threshold : float, default=0.1
        Threshold for binary classification metrics
    """
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 device: str = 'cpu', 
                 threshold: float = 0.1):
        self.model = model
        self.device = device
        self.metrics_calculator = PrecipitationMetrics(threshold=threshold)
        self.visualizer = MetricsVisualizer()
        
    def evaluate(self, 
                test_loader: torch.utils.data.DataLoader, 
                save_dir: Optional[Path] = None,
                thresholds: Optional[List[float]] = None,
                window_size: int = 5) -> Dict[str, float]:
        """
        Run complete evaluation on test dataset
        
        Parameters:
        -----------
        test_loader : DataLoader
            Test data loader
        save_dir : Path, optional
            Directory to save evaluation results
        thresholds : list of float, optional
            Additional thresholds for metrics
        window_size : int, default=5
            Window size for FSS calculation
            
        Returns:
        --------
        dict : Dictionary containing all calculated metrics
        """
        self.model.eval()
        self.model.to(self.device)
        
        all_predictions = []
        all_targets = []
        
        print("\n" + "="*60)
        print("RUNNING MODEL EVALUATION")
        print("="*60)
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                # Handle multi-output models
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        
        # Concatenate all batches
        y_pred = np.concatenate(all_predictions, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        
        print(f"\nTest dataset: {len(y_true)} samples")
        print(f"Prediction shape: {y_pred.shape}")
        print(f"Target shape: {y_true.shape}")
        
        # Calculate all metrics
        if thresholds is None:
            thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true, y_pred, 
            window_size=window_size,
            thresholds=thresholds
        )
        
        # Print metrics
        self.metrics_calculator.print_metrics(metrics)
        
        # Save results if directory provided
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            self.metrics_calculator.save_metrics(metrics, save_dir / 'evaluation_metrics.json')
            
            # Save metrics as DataFrame
            df = self.metrics_calculator.get_metrics_dataframe(metrics)
            df.to_csv(save_dir / 'evaluation_metrics.csv')
            
            # Save visualizations
            print("\nGenerating visualizations...")
            
            self.visualizer.plot_confusion_matrix(
                y_true, y_pred, 
                threshold=self.metrics_calculator.threshold,
                save_path=save_dir / 'confusion_matrix.png'
            )
            
            self.visualizer.plot_scatter_comparison(
                y_true, y_pred,
                title="Precipitation Prediction: Actual vs Predicted",
                save_path=save_dir / 'scatter_plot.png'
            )
            
            self.visualizer.plot_roc_curve(
                y_true, y_pred,
                save_path=save_dir / 'roc_curve.png'
            )
            
            print(f" All results saved to: {save_dir}")
        
        return metrics
    
    def compare_models(self, 
                      model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                      save_dir: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models' predictions
        
        Parameters:
        -----------
        model_predictions : dict
            Dictionary mapping model names to (y_true, y_pred) tuples
        save_dir : Path, optional
            Directory to save comparison results
            
        Returns:
        --------
        dict : Dictionary mapping model names to their metrics
        """
        all_metrics = {}
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        for model_name, (y_true, y_pred) in model_predictions.items():
            print(f"\nEvaluating {model_name}...")
            metrics = self.metrics_calculator.calculate_all_metrics(y_true, y_pred)
            all_metrics[model_name] = metrics
            self.metrics_calculator.print_metrics(metrics)
        
        # Plot comparison
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Classification metrics comparison
            metric_names = ['POD', 'FAR', 'CSI', 'HSS', 'GSS', 'F1_Score']
            self.visualizer.plot_metrics_comparison(
                all_metrics, metric_names,
                title="Precipitation Detection Metrics Comparison",
                save_path=save_dir / 'classification_comparison.png'
            )
            
            # Regression metrics comparison
            metric_names = ['RMSE', 'MAE', 'Correlation', 'KGE']
            self.visualizer.plot_metrics_comparison(
                all_metrics, metric_names,
                title="Regression Metrics Comparison",
                save_path=save_dir / 'regression_comparison.png'
            )
            
            print(f" Comparison results saved to: {save_dir}")
        
        return all_metrics


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def calculate_metrics_from_files(predictions_path: Union[str, Path], 
                                 targets_path: Union[str, Path],
                                 threshold: float = 0.1,
                                 save_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Calculate metrics from saved prediction and target files
    
    Parameters:
    -----------
    predictions_path : str or Path
        Path to predictions .npy file
    targets_path : str or Path
        Path to targets .npy file
    threshold : float
        Threshold for binary classification
    save_dir : Path, optional
        Directory to save results
        
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    y_pred = np.load(predictions_path)
    y_true = np.load(targets_path)
    
    metrics_calc = PrecipitationMetrics(threshold=threshold)
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred)
    
    metrics_calc.print_metrics(metrics)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics_calc.save_metrics(metrics, save_dir / 'metrics.json')
    
    return metrics


def create_metrics_report(metrics_dict: Dict[str, Dict[str, float]], 
                         output_path: Union[str, Path]) -> pd.DataFrame:
    """
    Create a comprehensive metrics report as CSV
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary mapping model names to their metrics
    output_path : str or Path
        Path to save the CSV report
        
    Returns:
    --------
    pd.DataFrame : Combined metrics DataFrame
    """
    all_dfs = []
    
    for model_name, metrics in metrics_dict.items():
        df = pd.DataFrame([metrics]).T
        df.columns = [model_name]
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, axis=1)
    combined_df.to_csv(output_path)
    
    print(f" Metrics report saved to: {output_path}")
    
    return combined_df


# ============================================================
# MAIN (for testing)
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("EVALUATION METRICS MODULE TEST")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    y_true = np.random.rand(100, 2, 128, 128) * 10
    y_pred = y_true + np.random.randn(100, 2, 128, 128) * 2
    
    # Initialize metrics calculator
    metrics_calc = PrecipitationMetrics(threshold=0.1)
    
    # Calculate all metrics
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred, thresholds=[0.1, 0.5, 1.0])
    
    # Print metrics
    metrics_calc.print_metrics(metrics)
    
    # Create visualizations
    visualizer = MetricsVisualizer()
    
    print("Generating test visualizations...")
    visualizer.plot_confusion_matrix(y_true, y_pred)
    visualizer.plot_scatter_comparison(y_true, y_pred)
    
    # Test metrics dataframe
    df = metrics_calc.get_metrics_dataframe(metrics)
    print("\nMetrics DataFrame:")
    print(df.head(10))
    
    print("\n" + "="*60)
    print(" All tests passed successfully!")
    print("="*60)