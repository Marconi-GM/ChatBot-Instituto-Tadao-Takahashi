# Use AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Copy and install dependencies first (for better caching)
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy the function code
COPY main.py ${LAMBDA_TASK_ROOT}
COPY prompts.yaml ${LAMBDA_TASK_ROOT}
COPY services/ ${LAMBDA_TASK_ROOT}/services/
COPY faiss_index/ ${LAMBDA_TASK_ROOT}/faiss_index/

# Tell Lambda which function to call
CMD ["main.handler"]
