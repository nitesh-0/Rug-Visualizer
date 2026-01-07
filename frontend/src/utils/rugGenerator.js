/**
 * Client-side rug pattern generator
 */

export const RUG_PATTERNS = [
  { id: 1, name: 'Modern Grid', pattern: 'grid', colors: ['#E8DFD5', '#C9A962', '#8B7355'] },
  { id: 2, name: 'Persian Classic', pattern: 'persian', colors: ['#8B2323', '#C9A962', '#2D2D2D'] },
  { id: 3, name: 'Boho Diamond', pattern: 'bohemian', colors: ['#C67B5C', '#8FA68A', '#C9A962'] },
  { id: 4, name: 'Minimal Stripe', pattern: 'stripe', colors: ['#F5F1EB', '#C67B5C', '#E8DFD5'] },
  { id: 5, name: 'Geometric', pattern: 'geometric', colors: ['#C9A962', '#E8DFD5', '#8B7355'] },
  { id: 6, name: 'Solid Sage', pattern: 'solid', colors: ['#8FA68A'] },
];

export function generateRugPattern(pattern, colors, width = 400, height = 280) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  
  // Background
  ctx.fillStyle = colors[0] || '#E8DFD5';
  ctx.fillRect(0, 0, width, height);
  
  switch (pattern) {
    case 'grid':
      ctx.fillStyle = colors[1] || '#C9A962';
      for (let x = 0; x < width; x += 50) {
        ctx.fillRect(x, 0, 20, height);
      }
      ctx.globalAlpha = 0.7;
      ctx.fillStyle = colors[2] || '#8B7355';
      for (let y = 0; y < height; y += 50) {
        ctx.fillRect(0, y, width, 20);
      }
      ctx.globalAlpha = 1;
      break;
      
    case 'persian':
      ctx.strokeStyle = colors[1] || '#C9A962';
      ctx.lineWidth = 12;
      ctx.strokeRect(12, 12, width - 24, height - 24);
      ctx.beginPath();
      ctx.ellipse(width/2, height/2, 80, 55, 0, 0, Math.PI * 2);
      ctx.fillStyle = colors[1];
      ctx.fill();
      ctx.strokeStyle = colors[2] || '#2D2D2D';
      ctx.lineWidth = 3;
      ctx.stroke();
      break;
      
    case 'bohemian':
      const size = 40;
      for (let y = -size; y < height + size; y += size) {
        for (let x = -size; x < width + size; x += size) {
          const offset = ((y / size) % 2) * (size / 2);
          ctx.beginPath();
          ctx.moveTo(x + offset + size/2, y);
          ctx.lineTo(x + offset + size, y + size/2);
          ctx.lineTo(x + offset + size/2, y + size);
          ctx.lineTo(x + offset, y + size/2);
          ctx.closePath();
          ctx.fillStyle = colors[Math.floor((x + y) / size) % colors.length];
          ctx.fill();
        }
      }
      break;
      
    case 'stripe':
      for (let y = 0; y < height; y += 20) {
        ctx.fillStyle = colors[(y / 20) % colors.length];
        ctx.fillRect(0, y, width, 20);
      }
      break;
      
    case 'geometric':
      const gridSize = 35;
      for (let y = 0; y < height; y += gridSize) {
        for (let x = 0; x < width; x += gridSize) {
          ctx.fillStyle = colors[((x / gridSize) + (y / gridSize)) % colors.length];
          ctx.fillRect(x + 3, y + 3, gridSize - 6, gridSize - 6);
        }
      }
      break;
      
    case 'solid':
      ctx.fillStyle = 'rgba(0,0,0,0.03)';
      for (let i = 0; i < 500; i++) {
        ctx.fillRect(Math.random() * width, Math.random() * height, 2, 2);
      }
      break;
  }
  
  // Border
  ctx.strokeStyle = 'rgba(0,0,0,0.15)';
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, width, height);
  
  return canvas.toDataURL();
}

export function generateAllPreviews() {
  const previews = {};
  RUG_PATTERNS.forEach(rug => {
    previews[rug.id] = generateRugPattern(rug.pattern, rug.colors);
  });
  return previews;
}