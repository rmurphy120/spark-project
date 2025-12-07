# Minimal Spark + PySpark dev environment
FROM eclipse-temurin:17-jdk

# Install tools: git (optional), Python 3, pip, Maven, build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        vim \
        ca-certificates \
        gnupg \
        bash-completion \
        maven \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Work in /opt/spark
WORKDIR /opt/spark

# Helpful env vars for PySpark
ENV SPARK_HOME=/opt/spark \
    PYSPARK_PYTHON=python3 \
    PYTHONUNBUFFERED=1

# Cache Maven dependencies (optional but nice for faster builds)
# Copy only dependency-related files so this layer doesn't get invalidated
COPY pom.xml .
COPY project ./project

RUN mvn -q -DskipTests dependency:go-offline || true

# Expose common Spark ports (optional; for UIs)
EXPOSE 4040 7077 8080 18080

# Default to a shell
CMD ["/bin/bash"]
