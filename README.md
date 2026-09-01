To create a postresql container in docker

docker run --name pgvector-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16
