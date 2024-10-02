import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from celery import Celery, chain
from celery.result import AsyncResult
import subprocess
from collections import deque

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# 確保上傳文件夾存在
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LLaVA 模型配置
LLAVA_CLI = "/Users/fuchangwei/llama.cpp/llama-llava-cli"
LLAVA_MODEL = "/Users/fuchangwei/LLaVA/llava-llama-3-8b-v1_1-gguf/llava-llama-3-8b-v1_1-int4.gguf"
MMPROJ_MODEL = "/Users/fuchangwei/LLaVA/llava-llama-3-8b-v1_1-gguf/llava-llama-3-8b-v1_1-mmproj-f16.gguf"
LLAVA_COMMAND = f"{LLAVA_CLI} -m {LLAVA_MODEL} --mmproj {MMPROJ_MODEL} --image {{}} -c 4096"

# 圖片處理隊列
image_queue = deque()

@celery.task(bind=True)
def process_image_task(self, image_path, filename):
    prompt = "<|start_header_id|>user<|end_header_id|>\n\n<image>\nDescribe the contents of this image, including the objects and their relative positions.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    command = LLAVA_COMMAND.format(image_path) + f' -e -p "{prompt}"'
    
    logger.info(f"Executing command for {filename}: {command}")
    
    try:
        if not os.path.exists(LLAVA_CLI):
            raise FileNotFoundError(f"llama-llava-cli not found at {LLAVA_CLI}")
        
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=300)
        output = result.stdout
        
        logger.info(f"Command output for {filename}: {output}")
        
        return {'filename': filename, 'result': output}
    
    except subprocess.TimeoutExpired:
        logger.error(f"Command execution timed out for {filename}")
        return {'filename': filename, 'error': 'Command execution timed out'}
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed for {filename}: {str(e)}")
        logger.error(f"Command output: {e.output}")
        return {'filename': filename, 'error': str(e)}
    except FileNotFoundError as e:
        logger.error(str(e))
        return {'filename': filename, 'error': str(e)}
    except Exception as e:
        logger.error(f"Unexpected error for {filename}: {str(e)}")
        return {'filename': filename, 'error': str(e)}

@celery.task
def process_next_image():
    if image_queue:
        next_image = image_queue.popleft()
        return process_image_task.s(next_image['path'], next_image['filename'])
    return None

@celery.task
def chain_next_task(result):
    next_task = process_next_image.s()
    next_task.apply_async()
    return result

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file')
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_queue.append({'path': filepath, 'filename': filename})
        
        # 開始處理第一張圖片
        if len(image_queue) == len(files):  # 確保所有文件都已加入隊列
            first_task = process_next_image.s()
            chain(first_task, chain_next_task.s()).apply_async()
        
        return jsonify({'message': f'{len(files)} files uploaded and queued for processing'})
    return render_template('upload.html')

@app.route('/status')
def status():
    processed = []
    pending = []
    
    # 檢查所有已知的任務狀態
    for task in AsyncResult.iterate_all():
        if task.state == 'SUCCESS':
            result = task.result
            if isinstance(result, dict) and 'filename' in result:
                processed.append({
                    'filename': result['filename'],
                    'status': 'Completed',
                    'result': result.get('result', ''),
                    'error': result.get('error', '')
                })
        elif task.state in ['PENDING', 'STARTED']:
            pending.append({
                'task_id': task.id,
                'status': task.state
            })
    
    # 添加仍在隊列中的圖片
    for image in image_queue:
        pending.append({
            'filename': image['filename'],
            'status': 'Queued'
        })
    
    return jsonify({
        'processed': processed,
        'pending': pending,
        'queue_length': len(image_queue)
    })

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)