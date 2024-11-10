import os 
import tensorflow as tf


def get_symmetric_quantize_matrix(x, target_dtype):
    """
    Function to quantize the weights using SYMMETRIC quantization.

    * Parameters:
        x: Tensor to be quantized
        target_dtype: Target data type for quantization

    * Returns:
        quantized: Quantized tensor
        scale: Scale factor
        zero_point: 0.0
    """

    # Define the range for target type
    qmin = tf.constant(target_dtype.min, dtype=tf.float32)
    qmax = tf.constant(target_dtype.max, dtype=tf.float32)

    # Compute the scale factor
    max_abs_val = tf.reduce_max(tf.abs(x))
    scale = max_abs_val / qmax
    zero_point = 0.0

    # Quantization and Clip to ensure quantized values are within the range
    quantized = tf.round(x / scale)  # Perform quantization
    quantized = tf.clip_by_value(quantized, qmin, qmax) 
    
    return tf.cast(quantized, target_dtype), scale, zero_point


def get_asymmetric_quantize_matrix(x, target_dtype):
    """
    Function to quantize the weights using asymmetric quantization.

    * Parameters:
        x: Tensor to be quantized
        target_dtype: Target data type for quantization

    * Returns:
        quantized: Quantized tensor
        scale: Scale factor
        zero_point: Zero point
    """

    # Define the range for the target type
    qmin = tf.constant(target_dtype.min, dtype=tf.float32)
    qmax = tf.constant(target_dtype.max, dtype=tf.float32)
    
    # Compute the min and max for the actual tensor
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)

    # Scale and zero point calculations
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = tf.cast(qmin - x_min / scale, tf.float32)

    # Quantization and Clip to ensure quantized values are within the range
    quantized = tf.round((x - x_min) / scale + qmin)
    quantized = tf.clip_by_value(quantized, qmin, qmax)

    return tf.cast(quantized, target_dtype), scale, zero_point