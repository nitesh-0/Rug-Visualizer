import React, { useRef, useEffect } from 'react';

export default function CanvasDisplay({ 
  compositedImage, 
  roomImage, 
  depthMap, 
  showDepthOverlay,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  canDrag
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const imgSrc = compositedImage || roomImage;
    
    if (imgSrc) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // Depth overlay
        if (showDepthOverlay && depthMap) {
          const depthImg = new Image();
          depthImg.onload = () => {
            ctx.globalAlpha = 0.4;
            ctx.drawImage(depthImg, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
          };
          depthImg.src = `data:image/png;base64,${depthMap}`;
        }
      };
      img.src = imgSrc;
    } else {
      // Empty state
      ctx.fillStyle = '#F5F1EB';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#A0A0A0';
      ctx.font = '16px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('Upload a room photo to begin', canvas.width / 2, canvas.height / 2);
    }
  }, [compositedImage, roomImage, showDepthOverlay, depthMap]);

  return (
    <canvas
      ref={canvasRef}
      width={1000}
      height={600}
      className={`w-full h-auto ${canDrag ? 'cursor-move' : 'cursor-default'}`}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    />
  );
}