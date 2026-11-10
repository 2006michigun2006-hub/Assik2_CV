import base64
import io
import json
import platform
import time
import warnings

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from matplotlib import cm
from flask import Flask, render_template, request

warnings.filterwarnings('ignore', message='xFormers')

CLASSES = ['bull', 'leon', 'shelly', 'spike', '8-bit', 'bibi', 'bo', 'emz', 'frank', 'miko']
THRESHOLD = 0.5
torch.set_grad_enabled(False)

prepare = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
model = nn.Sequential(backbone, nn.Linear(384, len(CLASSES)))
model.load_state_dict(torch.load('saved/dino_ft.pt', map_location='cpu'))
model.eval()

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
dino.eval()
features = torch.load('saved/dino_features.pt')
gallery_paths = []
gallery_x = []
for name in CLASSES:
    for i in range(1, 21):
        path = f'dataset/{name}/{name}_{i:03d}.jpg'
        gallery_paths.append(path)
        gallery_x.append(features[path])
gallery_x = nn.functional.normalize(torch.stack(gallery_x), dim=1)

saved = {}

def save_tokens(module, inputs, output):
    output.retain_grad()
    saved['tokens'] = output

backbone.blocks[-1].norm1.register_forward_hook(save_tokens)

with open('saved/cv_baselines.json') as f:
    cv = json.load(f)
with open('saved/cv_dino.json') as f:
    cv_dino = json.load(f)
metrics = []
for title, acc, f1 in [('Simple CNN (A1)', cv['cnn_acc'], cv['cnn_f1']),
                       ('ResNet18 (A1)', cv['resnet_acc'], cv['resnet_f1']),
                       ('DINOv2 kNN', cv_dino['knn_acc'], cv_dino['knn_f1']),
                       ('DINOv2 linear probe', cv_dino['linear_acc'], cv_dino['linear_f1']),
                       ('DINOv2 fine-tuned (this app)', cv_dino['ft_acc'], cv_dino['ft_f1'])]:
    metrics.append((title, f'{np.mean(acc):.3f} ± {np.std(acc):.3f}', f'{np.mean(f1):.3f} ± {np.std(f1):.3f}'))

latencies = []
info = {'device': 'CPU (' + platform.processor() + ')', 'threads': torch.get_num_threads(),
        'torch': torch.__version__, 'os': platform.system() + ' ' + platform.release(),
        'model': 'DINOv2 ViT-S/14, last 4 blocks fine-tuned'}


def crop_screenshot(img):
    w, h = img.size
    if w > 1.8 * h:
        s = h / 591
        x, y, size = int(430 * s), int(63 * s), int(504 * s)
        img = img.crop((x, y, x + size, y + size))
    return img


def to_data_url(img):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def grad_cam(x, label):
    with torch.enable_grad():
        out = model(x)
        model.zero_grad()
        out[0, label].backward()
    tokens = saved['tokens'][0, 1:]
    grads = saved['tokens'].grad[0, 1:]
    weights = grads.mean(0)
    cam = torch.relu((tokens * weights).sum(1)).reshape(16, 16)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy()


def analyze(img):
    img = crop_screenshot(img.convert('RGB'))
    x = prepare(img).unsqueeze(0)

    start = time.perf_counter()
    probs = torch.softmax(model(x), dim=1)[0]
    ms = (time.perf_counter() - start) * 1000
    latencies.append(ms)

    top = probs.topk(5)
    top5 = []
    for p, i in zip(top.values, top.indices):
        top5.append((CLASSES[i], f'{p.item() * 100:.1f}%'))
    best = top.indices[0].item()
    sure = top.values[0].item() >= THRESHOLD

    cam = grad_cam(x, best)
    cam = np.array(Image.fromarray(np.uint8(cam * 255)).resize((224, 224), Image.BILINEAR)) / 255
    heat = cm.jet(cam)[:, :, :3] * 255
    small = np.array(img.resize((224, 224)))
    overlay = Image.fromarray(np.uint8(0.5 * small + 0.5 * heat))

    query = nn.functional.normalize(dino(x), dim=1)
    similarity = (query @ gallery_x.T)[0]
    nearest = similarity.topk(6)
    neighbors = []
    for s, i in zip(nearest.values, nearest.indices):
        if s.item() > 0.999 or len(neighbors) == 5:
            continue
        path = gallery_paths[i]
        neighbors.append((to_data_url(Image.open(path)), path.split('/')[1], f'{s.item():.2f}'))

    return {'name': CLASSES[best] if sure else 'not sure (unknown brawler?)', 'sure': sure,
            'top5': top5, 'image': to_data_url(img.resize((224, 224))), 'heatmap': to_data_url(overlay),
            'neighbors': neighbors, 'ms': f'{ms:.0f}', 'mean_ms': f'{np.mean(latencies):.0f}', 'runs': len(latencies)}


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        file = request.files.get('photo')
        if file and file.filename:
            try:
                result = analyze(Image.open(file.stream))
            except OSError:
                error = 'This file is not a picture.'
    return render_template('index.html', result=result, error=error, info=info, metrics=metrics,
                           browser=request.headers.get('User-Agent'))


if __name__ == '__main__':
    app.run(port=3000)
