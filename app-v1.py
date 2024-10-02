import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import subprocess
from celery import Celery
import time

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
LLAVA_CLI = "/Users/fuchangwei/llama.cpp/llama-llava-cli"  # 替換為實際路徑
LLAVA_MODEL = "/Users/fuchangwei/LLaVA/llava-llama-3-8b-v1_1-gguf/llava-llama-3-8b-v1_1-int4.gguf"
MMPROJ_MODEL = "/Users/fuchangwei/LLaVA/llava-llama-3-8b-v1_1-gguf/llava-llama-3-8b-v1_1-mmproj-f16.gguf"
LLAVA_COMMAND = f"{LLAVA_CLI} -m {LLAVA_MODEL} --mmproj {MMPROJ_MODEL} --image {{}} -c 4096"

@celery.task(bind=True)
def process_image_task(self, image_path, filename):
    prompt = "<|start_header_id|>user<|end_header_id|>\n\n<image>\nDescribe the contents of this image, including the objects and their relative positions.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    command = LLAVA_COMMAND.format(image_path) + f' -e -p "{prompt}"'
    
    logger.info(f"Executing command: {command}")
    
    try:
        # 檢查文件是否存在
        if not os.path.exists(LLAVA_CLI):
            raise FileNotFoundError(f"llama-llava-cli not found at {LLAVA_CLI}")
        
        # 執行命令並捕獲輸出
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=300)  # 添加timeout
        output = result.stdout
        
        logger.info(f"Command output: {output}")
        
        # 返回結果
        return {'filename': filename, 'result': output}
    
    except subprocess.TimeoutExpired:
        logger.error("Command execution timed out")
        return {'filename': filename, 'error': 'Command execution timed out'}
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed: {str(e)}")
        logger.error(f"Command output: {e.output}")
        return {'filename': filename, 'error': str(e)}
    except FileNotFoundError as e:
        logger.error(str(e))
        return {'filename': filename, 'error': str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {'filename': filename, 'error': str(e)}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file')
        task_ids = []
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                task = process_image_task.apply_async(args=[filepath, filename])
                task_ids.append(task.id)
        return jsonify({'task_ids': task_ids})
    return render_template('upload.html')

@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = process_image_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'status': task.info.get('status', ''),
            'filename': task.info.get('filename', '')
        }
        if task.state == 'SUCCESS':
            response['result'] = task.result.get('result', '')
            if 'error' in task.result:
                response['error'] = task.result['error']
    else:
        response = {
            'state': task.state,
            'status': str(task.info),
        }
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)


