variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
  default     = "example-mlops-project"
}

variable "region" {
  description = "Google Cloud region for Cloud Run and Artifact Registry"
  type        = string
  default     = "europe-west3"
}

variable "container_image" {
  description = "Container image URI pushed to GHCR or Artifact Registry"
  type        = string
  default     = "ghcr.io/bisher-eng-1/cloud-native-ai-mlops-platform:latest"
}

variable "database_url" {
  description = "Neon Cloud PostgreSQL connection string"
  type        = string
  sensitive   = true
  default     = "postgresql://user:password@ep-example.neon.tech/neondb?sslmode=require"
}
