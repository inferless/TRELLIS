import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import uuid
from io import BytesIO
import base64
import requests
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import boto3


os.environ['SPCONV_ALGO'] = 'native'

class InferlessPythonModel:
    @staticmethod
    def download_image(url):
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert("RGB")
    
    @staticmethod
    def convert_base64(file_name):
        with open(file_name, 'rb') as file:
            file_content = file.read()
        base64_encoded = base64.b64encode(file_content)
        base64_string = base64_encoded.decode('utf-8')
        os.remove(file_name)
        return base64_string

    def initialize(self):
        model_id = "JeffreyXiang/TRELLIS-image-large"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id)
        self.pipeline.cuda()

        aws_region = 'us-east-1'  # e.g., 'us-west-1'
        self.s3_client = boto3.client('s3', region_name=aws_region, aws_access_key_id=os.getenv("AWS_KEYS"), aws_secret_access_key=os.getenv("AWS_SECRETS") )

    def infer(self, inputs):
        image_url = inputs["image_url"]
        seed =  int(inputs.get("seed",0))
        ss_guidance_strength =  float(inputs.get("ss_guidance_strength",7.5))
        ss_sampling_steps =  int(inputs.get("ss_sampling_steps",12))
        slat_guidance_strength =  float(inputs.get("slat_guidance_strength",3))
        slat_sampling_steps = int(inputs.get("slat_sampling_steps",12))
        glb_extraction_simplify = float(inputs.get("glb_extraction_simplify",0.0))
        glb_extraction_texture_size = int(inputs.get("glb_extraction_texture_size",1024))
        preprocess_image = bool(inputs.get("preprocess_image",False))

        image = InferlessPythonModel.download_image(image_url).resize((512, 512))
        # Run the pipeline
        outputs = self.pipeline.run(
            image,
            seed=seed,
            preprocess_image=preprocess_image,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

        # Render the outputs
        trial_id = uuid.uuid4()
        s3_bucket_name = 'infer-global-models'
        s3_key_prefix = 'videos/'  # Folder path in your S3 bucket (optional)

        video = render_utils.render_video(outputs['gaussian'][0])['color']        
        buffer = BytesIO()
        imageio.mimsave(buffer, video, fps=30, format='mp4')
        buffer.seek(0)  # Reset the buffer position
        # Upload to S3
        s3_key = f"{s3_key_prefix}{trial_id}_gs.mp4"  # S3 key for the video
        self.s3_client.upload_fileobj(buffer, s3_bucket_name, s3_key)
        key_gs = s3_key

        video = render_utils.render_video(outputs['radiance_field'][0])['color']
        buffer = BytesIO()
        imageio.mimsave(buffer, video, fps=30, format='mp4')
        buffer.seek(0)
        s3_key = f"{s3_key_prefix}{trial_id}_rf.mp4"  # S3 key for the video
        self.s3_client.upload_fileobj(buffer, s3_bucket_name, s3_key)
        key_rf = s3_key

        
        video = render_utils.render_video(outputs['mesh'][0])['normal']
        buffer = BytesIO()
        imageio.mimsave(buffer, video, fps=30, format='mp4')
        buffer.seek(0)
        s3_key = f"{s3_key_prefix}{trial_id}_mesh.mp4"  # S3 key for the video
        self.s3_client.upload_fileobj(buffer, s3_bucket_name, s3_key)
        key_mesh = s3_key

        # GLB files can be extracted from the outputs
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=glb_extraction_simplify,          
            texture_size=glb_extraction_texture_size,  
        )
        
        # glb.export(f"{trial_id}.glb")

        return {
            "gaussian_video": key_gs,
            "radiance_field_video": key_rf,
            "mesh_video": key_mesh,
        }

    def finalize(self):
        self.pipeline = None
