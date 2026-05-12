> ## Documentation Index
> Fetch the complete documentation index at: https://docs.siliconflow.com/llms.txt
> Use this file to discover all available pages before exploring further.

# 创建图片生成请求

> Creates an image response for the given prompt. The URL for the generated image is valid for one hour. Please make sure to download and store it promptly to avoid any issues due to URL expiration.



## OpenAPI

````yaml post /images/generations
openapi: 3.0.0
info:
  title: SiliconFlow API
  description: The SiliconFlow REST API
  version: 1.0.0
  contact:
    name: SiliconFlow Support
    url: https://www.siliconflow.com/
  license:
    name: MIT
    url: https://github.com/siliconflow-inc/siliconflow-api/blob/main/LICENSE
servers:
  - url: https://api.siliconflow.com/v1
security:
  - bearerAuth: []
paths:
  /images/generations:
    post:
      tags:
        - Image
      summary: Image Generation
      description: >-
        Creates an image response for the given prompt. The URL for the
        generated image is valid for one hour. Please make sure to download and
        store it promptly to avoid any issues due to URL expiration.
      operationId: ImageGeneration
      requestBody:
        content:
          application/json:
            schema:
              oneOf:
                - $ref: '#/components/schemas/FLUX.2-pro'
                - $ref: '#/components/schemas/FLUX.2-flex'
                - $ref: '#/components/schemas/Qwen-Image'
                - $ref: '#/components/schemas/Z-Image'
                - $ref: '#/components/schemas/FLUX.1-Kontext'
                - $ref: '#/components/schemas/FLUX.1-Kontext-dev'
                - $ref: '#/components/schemas/FLUX-1.1-pro'
                - $ref: '#/components/schemas/FLUX-1.1-pro-Ultra'
                - $ref: '#/components/schemas/FLUX.1-schnell'
                - $ref: '#/components/schemas/FLUX.1-dev'
      responses:
        '200':
          description: '200'
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ImagesGenerationResponse'
        '400':
          description: BadRequest
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BadRquestData'
        '401':
          description: Unauthorized
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UnauthorizedData'
        '404':
          description: NotFound
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/NotFoundData'
        '429':
          description: RateLimit
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RateLimitData'
        '503':
          description: Overloaded
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/OverloadedtData'
        '504':
          description: Timeout
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TimeoutData'
      deprecated: false
components:
  schemas:
    FLUX.2-pro:
      title: FLUX.2-pro
      type: object
      required:
        - model
        - prompt
      properties:
        model:
          type: string
          default: black-forest-labs/FLUX.2-pro
          enum:
            - black-forest-labs/FLUX.2-pro
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          example: A serene landscape with mountains and a lake at sunset
          description: The text prompt describing the image to generate.
        image_size:
          type: string
          title: Image size, format is [width]x[height]
          description: |
            Image resolution in "widthxheight" format. Supported resolutions:
              - "512x512" (1:1)
              - "768x1024" (3:4)
              - "1024x768" (4:3)
              - "576x1024" (9:16)
              - "1024x576" (16:9)
          enum:
            - 512x512
            - 768x1024
            - 1024x768
            - 576x1024
            - 1024x576
          default: 512x512
        seed:
          title: Seed
          type: integer
          minimum: 0
          maximum: 9999999999
          description: >-
            Random seed for reproducible generation. If not specified, a random
            seed will be used.
        output_format:
          description: Output format for the generated image.
          default: png
          type: string
          enum:
            - png
            - jpeg
    FLUX.2-flex:
      title: FLUX.2-flex
      type: object
      required:
        - model
        - prompt
      properties:
        model:
          type: string
          default: black-forest-labs/FLUX.2-flex
          enum:
            - black-forest-labs/FLUX.2-flex
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          example: A futuristic cityscape with flying cars and neon lights
          description: The text prompt describing the image to generate.
        image_size:
          type: string
          title: Image size, format is [width]x[height]
          description: |
            Image resolution in "widthxheight" format. Supported resolutions:
              - "512x512" (1:1)
              - "768x1024" (3:4)
              - "1024x768" (4:3)
              - "576x1024" (9:16)
              - "1024x576" (16:9)
          enum:
            - 512x512
            - 768x1024
            - 1024x768
            - 576x1024
            - 1024x576
          default: 512x512
        seed:
          title: Seed
          type: integer
          minimum: 0
          maximum: 9999999999
          description: >-
            Random seed for reproducible generation. If not specified, a random
            seed will be used.
        cfg:
          title: CFG Scale
          type: number
          description: >-
            CFG (Classifier-Free Guidance) is a technique that adjusts how
            closely generated outputs follow input prompts by balancing
            precision and creativity. This field is only applicable to
            Qwen/Qwen-Image models.For text generation scenarios, the CFG value
            must be greater than 1. The official configuration uses 50 steps
            with CFG 4.0. When CFG is set too small, it becomes nearly
            impossible to generate text.
          minimum: 0.1
          maximum: 20
        num_inference_steps:
          title: Number of Inference Steps
          type: integer
          minimum: 1
          maximum: 50
          default: 25
          description: >-
            Number of denoising steps. More steps generally produce higher
            quality images but take longer.
        output_format:
          description: Output format for the generated image.
          default: png
          type: string
          enum:
            - png
            - jpeg
    Qwen-Image:
      title: Qwen-Image
      type: object
      required:
        - model
        - prompt
      properties:
        model:
          type: string
          enum:
            - Qwen/Qwen-Image
            - Qwen/Qwen-Image-Edit
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          example: >-
            an island near sea, with seagulls, moon shining over the sea, light
            house, boats int he background, fish flying over the sea
        negative_prompt:
          title: Negative Prompt
          type: string
          description: negative prompt
        image_size:
          type: string
          title: Image size, format is  [width]x[height].
          description: >
            Image resolution in "widthxheight" format (Required). To ensure
            optimal quality, using the recommended values for your model is
            strongly advised.
              Recommended Values:  
                - "1328x1328" (1:1)
                - "1664x928" (16:9)
                - "928x1664" (9:16)
                - "1472x1140" (4:3)
                - "1140x1472" (3:4)
                - "1584x1056" (3:2)
                - "1056x1584" (2:3)
        batch_size:
          title: Number Images
          description: number of output images
          type: integer
          minimum: 1
          maximum: 4
          default: 1
        seed:
          title: Seed
          type: integer
          minimum: 0
          maximum: 9999999999
        num_inference_steps:
          title: Number Inference Steps
          description: number of inference steps
          type: integer
          minimum: 1
          maximum: 100
          default: 20
        guidance_scale:
          title: Guidance Scale
          description: >-
            This value is used to control the degree of match between the
            generated image and the given prompt. The higher the value, the more
            the generated image will tend to strictly match the text prompt. The
            lower the value, the more creative and diverse the generated image
            will be, potentially containing more unexpected elements.
          type: number
          minimum: 0
          maximum: 20
          default: 7.5
        cfg:
          title: CFG Scale
          type: number
          description: >-
            CFG (Classifier-Free Guidance) is a technique that adjusts how
            closely generated outputs follow input prompts by balancing
            precision and creativity. This field is only applicable to
            Qwen/Qwen-Image models.For text generation scenarios, the CFG value
            must be greater than 1. The official configuration uses 50 steps
            with CFG 4.0. When CFG is set too small, it becomes nearly
            impossible to generate text.
          minimum: 0.1
          maximum: 20
        image:
          $ref: '#/components/schemas/upload_image'
    Z-Image:
      title: Z-Image
      type: object
      required:
        - model
        - prompt
      properties:
        model:
          type: string
          enum:
            - Tongyi-MAI/Z-Image-Turbo
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          example: >-
            an island near sea, with seagulls, moon shining over the sea, light
            house, boats int he background, fish flying over the sea
        negative_prompt:
          title: Negative Prompt
          type: string
          description: negative prompt
        image_size:
          type: string
          title: Image size, format is  [width]x[height].
          description: >
            Image resolution in "widthxheight" format (Required). To ensure
            optimal quality, using the recommended values for your model is
            strongly advised.
              Recommended Values:  
                - "512x512" (1:1)
                - "768x1024" (3:4)
                - "1024x576" (16:9)
                - "576x1024" (9:16)
        seed:
          title: Seed
          type: integer
          minimum: 0
          maximum: 9999999999
    FLUX.1-Kontext:
      title: FLUX.1-Kontext
      type: object
      required:
        - model
        - prompt
      properties:
        model:
          type: string
          default: black-forest-labs/FLUX.1-Kontext-max
          enum:
            - black-forest-labs/FLUX.1-Kontext-max
            - black-forest-labs/FLUX.1-Kontext-pro
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          example: >-
            an island near sea, with seagulls, moon shining over the sea, light
            house, boats int he background, fish flying over the sea
        input_image:
          $ref: '#/components/schemas/upload_image'
          description: The image parameter is a required field.
        seed:
          title: Seed
          type: integer
          minimum: 0
          maximum: 9999999999
        aspect_ratio:
          type: string
          description: Aspect ratio of the image between 21:9 and 9:21
          example: '21:9'
        output_format:
          description: Output format for the generated image. Can be 'jpeg' or 'png'.
          default: png
          type: string
          enum:
            - png
            - jpeg
        prompt_upsampling:
          description: >-
            Whether to perform upsampling on the prompt. If active,
            automatically modifies the prompt for more creative generation.
          type: boolean
          example: false
          default: false
        safety_tolerance:
          description: >-
            Tolerance level for input and output moderation. Between 0 and 6, 0
            being most strict, 6 being least strict. Limit of 2 for Image to
            Image.
          type: integer
          minimum: 0
          maximum: 6
          example: 2
    FLUX.1-Kontext-dev:
      title: FLUX.1-Kontext-dev
      type: object
      required:
        - model
        - prompt
        - image
      properties:
        model:
          type: string
          default: black-forest-labs/FLUX.1-Kontext-dev
          enum:
            - black-forest-labs/FLUX.1-Kontext-dev
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          default: >-
            an island near sea, with seagulls, moon shining over the sea, light
            house, boats int he background, fish flying over the sea
        seed:
          title: Seed
          type: integer
          minimum: 0
          maximum: 9999999999
        prompt_enhancement:
          type: boolean
          description: >-
            Prompt enhancement switch, When enabled, the prompt is optimized to
            be detailed and model-friendly.
          example: false
          default: false
        image:
          $ref: '#/components/schemas/upload_image'
          description: The image parameter is a required field.
    FLUX-1.1-pro:
      title: FLUX-1.1-pro
      type: object
      properties:
        model:
          type: string
          default: black-forest-labs/FLUX-1.1-pro
          enum:
            - black-forest-labs/FLUX-1.1-pro
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          example: ein fantastisches bild
        image_prompt:
          type: string
          description: Optional base64 encoded image to use as a prompt for generation.
        width:
          type: integer
          description: Width of the generated image in pixels. It must be a multiple of 32.
          minimum: 256
          maximum: 1440
          default: 1024
        height:
          type: integer
          description: Width of the generated image in pixels. It must be a multiple of 32.
          minimum: 256
          maximum: 1440
          default: 768
        prompt_upsampling:
          type: boolean
          default: false
          description: >-
            Whether to upsample the prompt. If enabled, the prompt will be
            automatically adjusted to encourage more creative generation.
        seed:
          type: integer
          minimum: 0
          maximum: 9999999999
        safety_tolerance:
          type: integer
          description: >-
            Tolerance level for input and output review. Ranges from 0 to 6,
            where 0 is the strictest and 6 is the most lenient.
          minimum: 0
          maximum: 6
          default: 2
        output_format:
          type: string
          description: >-
            Output format for the generated image. It can be either 'jpeg' or
            'png'.
          enum:
            - jpeg
            - png
    FLUX-1.1-pro-Ultra:
      title: FLUX-1.1-pro-Ultra
      type: object
      properties:
        model:
          type: string
          default: black-forest-labs/FLUX-1.1-pro-Ultra
          enum:
            - black-forest-labs/FLUX-1.1-pro-Ultra
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          default: >-
            an island near sea, with seagulls, moon shining over the sea, light
            house, boats int he background, fish flying over the sea
        negative_prompt:
          title: Negative Prompt
          type: string
          description: negative prompt
        image_size:
          title: Image size, format is  [width]x[height].
          enum:
            - 1024x1024
            - 960x1280
            - 768x1024
            - 720x1440
            - 720x1280
            - others
          default: 1024x1024
        batch_size:
          title: Number Images
          description: number of output images
          type: integer
          minimum: 1
          maximum: 4
          default: 1
        seed:
          type: integer
          minimum: 0
          maximum: 9999999999
        aspect_ratio:
          type: string
          description: Aspect ratio of the image between 21:9 and 9:21
          example: '21:9'
        safety_tolerance:
          type: integer
          description: >-
            Tolerance level for input and output review. Ranges from 0 to 6,
            where 0 is the strictest and 6 is the most lenient.
          minimum: 0
          maximum: 6
          default: 2
        output_format:
          type: string
          description: >-
            Output format for the generated image. It can be either 'jpeg' or
            'png'.
          enum:
            - jpeg
            - png
        raw:
          type: boolean
          default: false
          description: Generate less processed, more natural-looking images
        image_prompt:
          type: string
          description: Optional image to remix in base64 format
        image_prompt_strength:
          type: integer
          description: Blend between the prompt and the image prompt
          minimum: 0
          maximum: 1
          default: 0.1
    FLUX.1-schnell:
      title: FLUX.1-schnell
      type: object
      required:
        - model
        - prompt
        - image_size
      properties:
        model:
          type: string
          default: black-forest-labs/FLUX.1-schnell
          enum:
            - black-forest-labs/FLUX.1-schnell
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          default: >-
            an island near sea, with seagulls, moon shining over the sea, light
            house, boats int he background, fish flying over the sea
        image_size:
          title: Image Size
          description: image size, format is [width]x[height]
          enum:
            - 1024x1024
            - 512x1024
            - 768x512
            - 768x1024
            - 1024x576
            - 576x1024
          default: 1024x1024
        seed:
          title: Seed
          type: integer
          minimum: 0
          maximum: 9999999999
        prompt_enhancement:
          type: boolean
          description: >-
            Prompt enhancement switch, When enabled, the prompt is optimized to
            be detailed and model-friendly.
          example: false
          default: false
    FLUX.1-dev:
      title: FLUX.1-dev
      type: object
      required:
        - model
        - prompt
        - image_size
        - num_inference_steps
      properties:
        model:
          type: string
          default: black-forest-labs/FLUX.1-dev
          enum:
            - black-forest-labs/FLUX.1-dev
          description: >-
            Corresponding Model Name. To better enhance service quality, we will
            make periodic changes to the models provided by this service,
            including but not limited to model on/offlining and adjustments to
            model service capabilities. We will notify you of such changes
            through appropriate means such as announcements or message pushes
            where feasible.
        prompt:
          type: string
          default: >-
            an island near sea, with seagulls, moon shining over the sea, light
            house, boats int he background, fish flying over the sea
        image_size:
          title: >-
            Image size, format is  [width]x[height], with a maximum of 2359296
            pixels.
          enum:
            - 1024x1024
            - 960x1280
            - 768x1024
            - 720x1440
            - 720x1280
            - others
          default: 1024x1024
        seed:
          title: Seed
          type: integer
          minimum: 0
          maximum: 9999999999
        num_inference_steps:
          title: Number Inference Steps
          description: inference steps
          type: integer
          minimum: 1
          maximum: 30
          default: 20
        prompt_enhancement:
          type: boolean
          description: >-
            Prompt enhancement switch, When enabled, the prompt is optimized to
            be detailed and model-friendly.
          example: false
          default: false
    ImagesGenerationResponse:
      type: object
      properties:
        images:
          type: array
          items:
            type: object
            properties:
              url:
                description: >-
                  The URL for the generated image is valid for one hour. Please
                  make sure to download and store it promptly to avoid any
                  issues due to URL expiration.
                type: string
        timings:
          type: object
          properties:
            inference:
              type: number
              format: float
        seed:
          type: integer
    BadRquestData:
      type: object
      required:
        - message
        - data
        - code
      properties:
        code:
          type: integer
          nullable: true
          default: false
          example: 20012
        message:
          type: string
          nullable: false
        data:
          type: string
          nullable: false
    UnauthorizedData:
      type: string
      default: false
      example: Invalid token
    NotFoundData:
      type: string
      default: false
      example: 404 page not found
    RateLimitData:
      type: object
      required:
        - message
        - data
      properties:
        message:
          type: string
          example: >-
            Request was rejected due to rate limiting. If you want more, please
            contact contact@siliconflow.com. Details:TPM limit reached.
        data:
          type: string
    OverloadedtData:
      type: object
      required:
        - code
        - message
        - data
      properties:
        code:
          type: integer
          example: 50505
        message:
          type: string
          example: Model service overloaded. Please try again later.
        data:
          type: string
          nullable: false
    TimeoutData:
      type: string
    upload_image:
      title: Upload Image
      description: >-
        The image that needs to be uploaded should be converted into base64
        format like "data:image/png;base64, XXX"
      type: string
      example: data:image/png;base64, XXX
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: your api key
      description: >-
        Use the following format for authentication: Bearer [<your api
        key>](https://cloud.siliconflow.com/account/ak)

````