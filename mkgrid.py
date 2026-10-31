"""Contact sheet of CAPTCHA images for hand labelling (dev helper)."""
import sys
from PIL import Image, ImageDraw
files = [l.strip() for l in open(sys.argv[1]) if l.strip()]
out_path = sys.argv[2]
S, cols = 3, 3
w, h = 150, 50
rows = (len(files) + cols - 1) // cols
out = Image.new('RGB', (cols * (w * S + 10), rows * (h * S + 22)), 'white')
d = ImageDraw.Draw(out)
for i, f in enumerate(files):
    im = Image.open(f).convert('RGBA')
    bg = Image.new('RGBA', im.size, 'white'); bg.alpha_composite(im)
    x = (i % cols) * (w * S + 10); y = (i // cols) * (h * S + 22)
    out.paste(bg.convert('RGB').resize((w * S, h * S), Image.LANCZOS), (x, y + 20))
    d.text((x + 2, y + 4), f'#{i}', fill='red')
out.save(out_path)
