import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import ffmpeg
import gradio as gr
from tqdm import tqdm
import zstandard as zstd
import brotli
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
import cupy as cp
import io
import mimetypes
from pydub import AudioSegment
from PyPDF2 import PdfFileReader, PdfFileWriter
import docx
import openpyxl

class GPUAcceleratedCompressionToolkit:
    def __init__(self):
        self.supported_formats = {
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            'video': ['.mp4', '.avi', '.mov', '.mkv'],
            'audio': ['.mp3', '.wav', '.ogg', '.flac'],
            'document': ['.txt', '.pdf', '.doc', '.docx'],
            'spreadsheet': ['.xlsx', '.xls', '.csv']
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect_file_type(self, file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            type_category = mime_type.split('/')[0]
            if type_category in ['image', 'video', 'audio']:
                return type_category
            elif type_category == 'application':
                if mime_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    return 'document'
                elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                    return 'spreadsheet'
        return 'other'

    def compress_image(self, input_path, output_path, compression_level, use_gpu=False, output_format='original'):
        img = Image.open(input_path)
        original_format = img.format if img.format else 'JPEG'
        
        if output_format == 'original':
            save_format = original_format
        else:
            save_format = output_format.upper()
        
        quality = int(compression_level)
        
        if use_gpu:
            tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            
            compressed = F.interpolate(tensor, scale_factor=quality/100, mode='bilinear', align_corners=False)
            compressed = F.interpolate(compressed, size=tensor.shape[2:], mode='bilinear', align_corners=False)
            
            result = transforms.ToPILImage()(compressed.squeeze(0).cpu())
            result.save(output_path, format=save_format, quality=quality)
        else:
            img.save(output_path, format=save_format, quality=quality)

    def compress_video(self, input_path, output_path, compression_level, use_gpu=False, output_format=None):
        if use_gpu:
            vcodec = 'h264_nvenc'
        else:
            vcodec = 'libx264'
        
        crf = int(100 - compression_level)  # Invert the scale for CRF
        
        if output_format is None:
            output_format = os.path.splitext(input_path)[1][1:]
        
        (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec=vcodec, crf=str(crf), acodec='aac', **{'preset': 'slow'})
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

    def compress_audio(self, input_path, output_path, compression_level, output_format=None):
        if output_format is None:
            output_format = os.path.splitext(input_path)[1][1:]
        
        bitrate = f"{int(compression_level * 3.2)}k"  # Scale compression_level to bitrate
        
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format=output_format, bitrate=bitrate)

    def compress_document(self, input_path, output_path, compression_level, output_format=None):
        if output_format is None:
            output_format = os.path.splitext(input_path)[1][1:]
        
        if output_format == 'pdf':
            with open(input_path, 'rb') as file:
                reader = PdfFileReader(file)
                writer = PdfFileWriter()
                for page in range(reader.getNumPages()):
                    page = reader.getPage(page)
                    page.compressContentStreams()  # This is CPU intensive!
                    writer.addPage(page)
                with open(output_path, 'wb') as output_file:
                    writer.write(output_file)
        elif output_format in ['doc', 'docx']:
            doc = docx.Document(input_path)
            doc.save(output_path)
        else:
            # For other document types, use generic file compression
            self.compress_file_gpu(input_path, output_path, compression_level)

    def compress_spreadsheet(self, input_path, output_path, compression_level, output_format=None):
        if output_format is None:
            output_format = os.path.splitext(input_path)[1][1:]
        
        wb = openpyxl.load_workbook(input_path)
        wb.save(output_path)

    def compress_file_gpu(self, input_path, output_path, compression_level):
        level = int(compression_level / 10)  # Scale compression_level to Zstandard level
        
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
        
        d_data = cp.asarray(bytearray(data))
        cctx = zstd.ZstdCompressor(level=level)
        d_compressed = cp.asarray(bytearray(cctx.compress(d_data.get())))
        compressed = d_compressed.get().tobytes()
        
        with open(output_path, 'wb') as f_out:
            f_out.write(compressed)

    def batch_compress_gpu(self, input_files, output_dir, use_gpu, output_format, compression_level):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = []
        for file in tqdm(input_files):
            input_path = file.name
            file_type = self.detect_file_type(input_path)
            
            # Determine output format and path
            if output_format == 'original':
                _, ext = os.path.splitext(input_path)
                output_path = os.path.join(output_dir, f"compressed_{os.path.basename(input_path)}")
            else:
                output_path = os.path.join(output_dir, f"compressed_{os.path.splitext(os.path.basename(input_path))[0]}.{output_format}")
            
            if file_type == 'image':
                self.compress_image(input_path, output_path, compression_level, use_gpu=use_gpu, output_format=output_format)
            elif file_type == 'video':
                self.compress_video(input_path, output_path, compression_level, use_gpu=use_gpu, output_format=output_format if output_format != 'original' else None)
            elif file_type == 'audio':
                self.compress_audio(input_path, output_path, compression_level, output_format=output_format if output_format != 'original' else None)
            elif file_type == 'document':
                self.compress_document(input_path, output_path, compression_level, output_format=output_format if output_format != 'original' else None)
            elif file_type == 'spreadsheet':
                self.compress_spreadsheet(input_path, output_path, compression_level, output_format=output_format if output_format != 'original' else None)
            else:
                self.compress_file_gpu(input_path, output_path, compression_level)
            
            results.append(output_path)
        
        return results

    def real_time_preview_gpu(self, input_path, compression_level, use_gpu=False):
        file_type = self.detect_file_type(input_path)
        
        if file_type == 'image':
            img = Image.open(input_path)
            if use_gpu:
                tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
                tensor = F.interpolate(tensor, size=(300, 300), mode='bilinear', align_corners=False)
                compressed = F.interpolate(tensor, scale_factor=compression_level/100, mode='bilinear', align_corners=False)
                compressed = F.interpolate(compressed, size=tensor.shape[2:], mode='bilinear', align_corners=False)
                result = transforms.ToPILImage()(compressed.squeeze(0).cpu())
            else:
                img = img.resize((300, 300))
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=int(compression_level))
                buffer.seek(0)
                result = Image.open(buffer)
            return result
        elif file_type == 'video':
            video = cv2.VideoCapture(input_path)
            ret, frame = video.read()
            if ret:
                if use_gpu:
                    d_frame = cp.asarray(frame)
                    d_frame = cp.resize(d_frame, (300, 300))
                    _, d_buffer = cv2.imencode('.jpg', cp.asnumpy(d_frame), [cv2.IMWRITE_JPEG_QUALITY, int(compression_level)])
                else:
                    frame = cv2.resize(frame, (300, 300))
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, int(compression_level)])
                return Image.open(io.BytesIO(buffer.tobytes()))
        elif file_type in ['audio', 'document', 'spreadsheet', 'other']:
            placeholder = Image.new('RGB', (300, 300), color='lightgray')
            draw = ImageDraw.Draw(placeholder)
            draw.text((10, 150), f"{file_type.capitalize()} Preview\nNot Available", fill='black')
            return placeholder
        
        return None

def gradio_interface(toolkit):
    def process_files(files, use_gpu, output_format, compression_level):
        output_dir = "compressed_output"
        return toolkit.batch_compress_gpu(files, output_dir, use_gpu, output_format, compression_level)
    
    def update_preview(file, compression_level, use_gpu):
        if file is None:
            return None
        return toolkit.real_time_preview_gpu(file.name, compression_level, use_gpu)

    iface = gr.Interface(
        fn=process_files,
        inputs=[
            gr.File(label="Input Files", file_count="multiple"),
            gr.Checkbox(label="Use GPU Acceleration"),
            gr.Dropdown(
                choices=["original", "jpg", "png", "mp4", "mp3", "pdf", "docx", "xlsx"],
                label="Output Format",
                value="original"
            ),
            gr.Slider(1, 100, 50, step=1, label="Compression Level")
        ],
        outputs=gr.File(label="Compressed Files", file_count="multiple"),
        title="GPU-Accelerated Compression Toolkit",
        description="Drag and drop files for compression of images, videos, audio, documents, spreadsheets, and other files. File type is automatically detected.",
        allow_flagging="never"
    )

    preview = gr.Interface(
        fn=update_preview,
        inputs=[
            gr.File(label="Input File"),
            gr.Slider(1, 100, 50, step=1, label="Compression Level"),
            gr.Checkbox(label="Use GPU Acceleration")
        ],
        outputs=gr.Image(label="Preview"),
        title="Real-time Compression Preview",
        live=True,
        allow_flagging="never"
    )

    return gr.TabbedInterface([iface, preview], ["Compress", "Preview"])

if __name__ == "__main__":
    toolkit = GPUAcceleratedCompressionToolkit()
    interface = gradio_interface(toolkit)
    interface.launch(share=True)