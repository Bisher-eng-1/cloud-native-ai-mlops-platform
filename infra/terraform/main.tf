terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "mlops_repo" {
  location      = var.region
  repository_id = "mlops-inference"
  description   = "Container registry for ONNX inference API"
  format        = "DOCKER"
}

resource "google_cloud_run_v2_service" "inference_api" {
  name     = "mlops-inference-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }

        cpu_idle = true
      }

      env {
        name  = "DATABASE_URL"
        value = var.database_url
      }

      env {
        name  = "MODEL_PATH"
        value = "artifacts/model_int8.onnx"
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        period_seconds        = 5
      }
    }
  }
}
