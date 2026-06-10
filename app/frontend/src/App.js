import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [processingStage, setProcessingStage] = useState('');
    const [pollingId, setPollingId] = useState(null);
    const [previewUrl, setPreviewUrl] = useState('');
    const [generatedVideoPreviewUrl, setGeneratedVideoPreviewUrl] = useState('');
    const [error, setError] = useState('');
    const [isDragOver, setIsDragOver] = useState(false);
    const fileInputRef = useRef(null);
    
    const backendBaseUrl = 'http://localhost:5000';

    // 组件卸载时清理资源
    useEffect(() => {
        return () => {
            if (pollingId) clearTimeout(pollingId);
            if (previewUrl) URL.revokeObjectURL(previewUrl);
            if (generatedVideoPreviewUrl) URL.revokeObjectURL(generatedVideoPreviewUrl);
        };
    }, [pollingId, previewUrl, generatedVideoPreviewUrl]);

    // 检查状态
    const checkStatus = async (videoId) => {
        try {
            const response = await axios.get(`${backendBaseUrl}/status/${videoId}`);

            if (response.data.status === 'completed') {
                setResult({
                    video_id: videoId,
                    download_url: `${backendBaseUrl}${response.data.download_url}`,
                    duration: response.data.duration,
                    file_size: response.data.file_size
                });
                setGeneratedVideoPreviewUrl(`${backendBaseUrl}${response.data.download_url}`);
                setIsLoading(false);
                setProcessingStage('完成！');
            } else {
                setProcessingStage('正在分析视频内容...');
                const id = setTimeout(() => checkStatus(videoId), 2000);
                setPollingId(id);
            }
        } catch (error) {
            console.error('状态检查失败:', error);
            setError('处理失败，请重试');
            setIsLoading(false);
        }
    };

    // 文件验证
    const validateFile = (file) => {
        const maxSize = 1024 * 1024 * 1024; // 1GB
        const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime'];
        
        if (file.size > maxSize) {
            return '文件大小超过1GB限制';
        }
        
        if (!allowedTypes.includes(file.type)) {
            return '不支持的文件格式，请上传MP4、AVI或MOV格式';
        }
        
        return null;
    };

    // 处理文件上传
    const handleUpload = async (e) => {
        e.preventDefault();
        if (!file) return;

        const validationError = validateFile(file);
        if (validationError) {
            setError(validationError);
            return;
        }

        setIsLoading(true);
        setError('');
        setUploadProgress(0);
        setProcessingStage('正在上传视频...');

        try {
            const formData = new FormData();
            formData.append('video', file);

            const uploadResponse = await axios.post(
                `${backendBaseUrl}/upload`,
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    },
                    onUploadProgress: (progressEvent) => {
                        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                        setUploadProgress(progress);
                        
                        if (progress === 100) {
                            setProcessingStage('上传完成，开始处理...');
                        }
                    }
                }
            );

            if (uploadResponse.data.video_id) {
                setProcessingStage('正在分析射门场景...');
                checkStatus(uploadResponse.data.video_id);
            }
        } catch (error) {
            console.error('上传失败:', error);
            setError(error.response?.data?.error || '上传失败，请检查网络连接');
            setIsLoading(false);
            setUploadProgress(0);
            setProcessingStage('');
        }
    };

    // 处理文件选择
    const handleFileChange = (selectedFile) => {
        if (!selectedFile) return;
        
        const validationError = validateFile(selectedFile);
        if (validationError) {
            setError(validationError);
            return;
        }

        setFile(selectedFile);
        setError('');
        
        // 创建预览URL
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        const url = URL.createObjectURL(selectedFile);
        setPreviewUrl(url);
        
        // 清理之前的结果
        setResult(null);
        if (generatedVideoPreviewUrl) {
            URL.revokeObjectURL(generatedVideoPreviewUrl);
            setGeneratedVideoPreviewUrl('');
        }
    };

    // 拖拽处理
    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragOver(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragOver(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragOver(false);
        
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) {
            handleFileChange(droppedFile);
        }
    };

    // 点击上传区域 - Fixed the linting issue
    const handleUploadAreaClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    // 重置表单
    const resetForm = () => {
        setFile(null);
        setResult(null);
        setError('');
        setUploadProgress(0);
        setProcessingStage('');
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        if (generatedVideoPreviewUrl) URL.revokeObjectURL(generatedVideoPreviewUrl);
        setPreviewUrl('');
        setGeneratedVideoPreviewUrl('');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    // 格式化时长
    const formatDuration = (duration) => {
        const minutes = Math.floor(duration / 60);
        const seconds = Math.floor(duration % 60).toString().padStart(2, '0');
        return `${minutes}:${seconds}`;
    };

    // 格式化文件大小
    const formatFileSize = (size) => {
        if (size < 1024 * 1024) {
            return (size / 1024).toFixed(1) + 'KB';
        }
        return (size / 1024 / 1024).toFixed(1) + 'MB';
    };

    return (
        <div className="container">
            <div className="upload-card">
                <div className="header">
                    <h1><span role="img" aria-label="足球">⚽</span> 足球射门集锦生成器</h1>
                    <p className="description">
                        使用AI智能识别足球比赛中的射门场景，自动生成精彩集锦视频
                    </p>
                    <div className="features">
                        <span className="feature"><span role="img" aria-label="目标">🎯</span> 智能识别</span>
                        <span className="feature"><span role="img" aria-label="闪电">⚡</span> 快速处理</span>
                        <span className="feature"><span role="img" aria-label="手机">📱</span> 移动友好</span>
                    </div>
                </div>

                {error && (
                    <div className="error-message">
                        <span className="error-icon" role="img" aria-label="警告">⚠️</span>
                        {error}
                        <button className="error-close" onClick={() => setError('')}>×</button>
                    </div>
                )}

                <form onSubmit={handleUpload} className="upload-form">
                    <div
                        className={`file-drop-zone ${isDragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={handleUploadAreaClick}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            onChange={(e) => handleFileChange(e.target.files[0])}
                            accept="video/mp4,video/avi,video/mov,video/quicktime"
                            className="file-input"
                        />

                        {file ? (
                            <div className="file-selected">
                                <div className="file-icon"><span role="img" aria-label="电影摄像机">🎬</span></div>
                                <div className="file-details">
                                    <div className="file-name">{file.name}</div>
                                    <div className="file-size">{formatFileSize(file.size)}</div>
                                </div>
                                <button
                                    type="button"
                                    className="file-remove"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        resetForm();
                                    }}
                                >
                                    ×
                                </button>
                            </div>
                        ) : (
                            <div className="file-placeholder">
                                <div className="upload-icon"><span role="img" aria-label="文件夹">📁</span></div>
                                <div className="upload-text">
                                    <strong>点击选择</strong> 或 <strong>拖拽视频文件</strong> 到此处
                                </div>
                                <div className="upload-hint">
                                    支持格式: MP4, AVI, MOV | 最大: 1GB
                                </div>
                            </div>
                        )}
                    </div>

                    {isLoading && (
                        <div className="processing-status">
                            <div className="progress-container">
                                <div className="progress-bar">
                                    <div
                                        className="progress-fill"
                                        style={{ width: `${uploadProgress}%` }}
                                    ></div>
                                </div>
                                <div className="progress-text">{uploadProgress}%</div>
                            </div>
                            <div className="processing-stage">
                                <span className="spinner"></span>
                                {processingStage}
                            </div>
                        </div>
                    )}

                    <button
                        type="submit"
                        className="generate-btn"
                        disabled={isLoading || !file}
                    >
                        {isLoading ? (
                            <>
                                <span className="spinner"></span>
                                处理中...
                            </>
                        ) : (
                            <>
                                <span className="btn-icon"><span role="img" aria-label="火箭">🚀</span></span>
                                开始生成集锦
                            </>
                        )}
                    </button>
                </form>

                {previewUrl && !isLoading && (
                    <div className="preview-section">
                        <h3><span role="img" aria-label="胶片">🎞️</span> 原始视频预览</h3>
                        <div className="video-wrapper">
                            <video controls className="preview-video" preload="metadata">
                                <source src={previewUrl} type="video/mp4" />
                                您的浏览器不支持视频播放
                            </video>
                        </div>
                    </div>
                )}

                {result && (
                    <div className="result-section">
                        <div className="success-header">
                            <div className="success-icon"><span role="img" aria-label="庆祝">🎉</span></div>
                            <div className="success-text">
                                <h3>集锦生成成功！</h3>
                                <div className="result-stats">
                                    <span className="stat">
                                        <span role="img" aria-label="时钟">⏱️</span> {formatDuration(result.duration)}
                                    </span>
                                    <span className="stat">
                                        <span role="img" aria-label="包裹">📦</span> {formatFileSize(result.file_size)}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {generatedVideoPreviewUrl && (
                            <div className="generated-preview">
                                <h4><span role="img" aria-label="电影摄像机">🎬</span> 生成的集锦预览</h4>
                                <div className="video-wrapper">
                                    <video controls className="preview-video" preload="metadata">
                                        <source src={generatedVideoPreviewUrl} type="video/mp4" />
                                        您的浏览器不支持视频播放
                                    </video>
                                </div>
                            </div>
                        )}

                        <div className="action-buttons">
                            <a
                                href={result.download_url}
                                className="download-btn primary"
                                download={`highlight_${result.video_id}.mp4`}
                            >
                                <span className="btn-icon"><span role="img" aria-label="下载">⬇️</span></span>
                                下载集锦视频
                            </a>
                            <button
                                className="download-btn secondary"
                                onClick={resetForm}
                            >
                                <span className="btn-icon"><span role="img" aria-label="刷新">🔄</span></span>
                                处理新视频
                            </button>
                        </div>
                    </div>
                )}
            </div>

            <div className="footer">
                <p><span role="img" aria-label="灯泡">💡</span> 提示：为了获得最佳效果，请上传包含明显射门场景的足球比赛视频</p>
            </div>
        </div>
    );
}

export default App;