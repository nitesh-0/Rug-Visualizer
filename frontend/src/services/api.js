/**
 * API Service for Rug Visualizer
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class APIService {
  async uploadRoom(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE}/upload/room`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error('Upload failed');
    }
    
    return response.json();
  }
  
  async analyzeRoom(imageId) {
    const formData = new FormData();
    formData.append('image_id', imageId);
    
    const response = await fetch(`${API_BASE}/analyze/complete`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error('Analysis failed');
    }
    
    return response.json();
  }
  
  async compositeRug(params) {
    const response = await fetch(`${API_BASE}/composite/realtime`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    
    if (!response.ok) {
      throw new Error('Compositing failed');
    }
    
    return response.json();
  }
}

export default new APIService();