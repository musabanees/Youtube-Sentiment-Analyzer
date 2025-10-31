# 1. Use a small, specific base image
FROM python:3.12-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=flask_app.app
ENV FLASK_ENV=production
ENV MPLCONFIGDIR=/tmp/matplotlib  

# 3. Set working directory
WORKDIR /app

# 4. Create unprivileged user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 5. Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 6. Copy ONLY dependency files first (leverage Docker cache)
COPY pyproject.toml README.md* ./

# 7. Install Python dependencies
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir gunicorn

# 8. Copy runtime files
COPY --chown=appuser:appgroup flask_app/ flask_app/
COPY --chown=appuser:appgroup params.yaml ./
COPY --chown=appuser:appgroup models/tfidf_vectorizer.pkl models/

# 9. Switch to unprivileged user
USER appuser

# 10. Expose port
EXPOSE 5000

# 11. Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# 12. Run with Gunicorn (production)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--threads", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "flask_app.app:app"]
