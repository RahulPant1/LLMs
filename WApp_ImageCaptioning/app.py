from flask import Flask, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename
import os
import ollama
import base64
from PIL import Image
import io
from flask import Flask, request, render_template, url_for, redirect, send_from_directory
from flask_socketio import SocketIO, emit


app = Flask(__name__)
socketio = SocketIO(app)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image if it's too large
        max_size = (800, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

def generate_funny_comment(image_path):
    try:
        # Convert image to base64
        base64_image = image_to_base64(image_path)
        
        # Prepare the prompt
        prompt = """Look at this image and:
        1. Tell me what you see in a brief description
        2. Make a funny, witty comment or joke about what's in the image
        Be creative and humorous but keep it friendly!"""
        
        # Make request to Ollama using llava model
        response = ollama.generate(
            model='llava',  # Using llava model which is designed for image understanding
            prompt=prompt,
            images=[base64_image],
            stream=False
        )
        
        if not response or 'response' not in response:
            return "Oops! Something went wrong while generating the comment. Maybe the image was too awesome for words! 😄"
            
        return response['response']
    except Exception as e:
        print(f"Error generating comment: {str(e)}")
        return f"Oops! An error occurred: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If no file selected
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(file_path)
            
            # Emit socket event to generate funny comment
            socketio.start_background_task(generate_comment_task, file_path, filename)
            
            return render_template('display_image.html', filename=filename)
    
    return render_template('index.html')

def generate_comment_task(file_path, filename):
    comment = generate_funny_comment(file_path)
    socketio.emit('comment_generated', {'filename': filename, 'comment': comment})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    socketio.run(app, debug=True,port=5011)