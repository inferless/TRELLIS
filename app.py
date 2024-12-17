import uuid
import os
from io import BytesIO
import base64
import requests
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

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
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.cuda()

    def infer(self, inputs):
        image_url = inputs["image_url"]
        seed =  inputs.get("seed",0)
        ss_guidance_strength =  inputs.get("ss_guidance_strength",7.5)
        ss_sampling_steps =  inputs.get("ss_sampling_steps",12)
        slat_guidance_strength =  inputs.get("slat_guidance_strength",3)
        slat_sampling_steps = inputs.get("slat_sampling_steps",12)
        glb_extraction_simplify = inputs.get("glb_extraction_simplify",0.95)
        glb_extraction_texture_size = inputs.get("glb_extraction_texture_size",1024)
        preprocess_image = inputs.get("preprocess_image",False)

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
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(f"{trial_id}_gs.mp4", video, fps=30)
        video = render_utils.render_video(outputs['radiance_field'][0])['color']
        imageio.mimsave(f"{trial_id}_rf.mp4", video, fps=30)
        video = render_utils.render_video(outputs['mesh'][0])['normal']
        imageio.mimsave(f"{trial_id}_mesh.mp4", video, fps=30)

        # GLB files can be extracted from the outputs
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=glb_extraction_simplify,          
            texture_size=glb_extraction_texture_size,  
        )
        glb.export(f"{trial_id}.glb")

        return {
            "gaussian_video": InferlessPythonModel.convert_base64(f"{trial_id}_gs.mp4"),
            "radiance_field_video": InferlessPythonModel.convert_base64(f"{trial_id}_rf.mp4"),
            "mesh_video": InferlessPythonModel.convert_base64(f"{trial_id}_mesh.mp4"),
            "GLB": InferlessPythonModel.convert_base64(f"{trial_id}.glb")
        }

    def finalize(self):
        self.pipeline = None