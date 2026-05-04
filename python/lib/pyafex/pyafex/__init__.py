"""Python bindings for the afex library.

analyze_audio_file() and analyze_audio_file_with_yaml() accept an optional
trim_silence=True flag to remove leading and trailing silence before feature
analysis. They also accept mutually exclusive max_frame_size and
max_frame_length frame limits, where max_frame_length is in seconds.
"""

from ._pyafex import analyze_audio_file, analyze_audio_file_with_yaml, builtin_feature_names, placeholder_method

__all__ = ["analyze_audio_file", "analyze_audio_file_with_yaml", "builtin_feature_names", "placeholder_method"]
