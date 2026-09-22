variable "yc_token" {
  type      = string
  sensitive = true
}

variable "yc_cloud_id" {
  type = string
}

variable "yc_folder_id" {
  type = string
}

variable "zone" {
  type    = string
  default = "ru-central1-a"
}

variable "vm_name" {
  type    = string
  default = "basic-rl-model-vm-3"
}

variable "ssh_public_key" {
  type = string
}

variable "platform_id" {
  type    = string
  default = "standard-v3"
}

variable "cores" {
  type    = number
  default = 8
}

variable "memory" {
  type    = number
  default = 128
}

variable "core_fraction" {
  type    = number
  default = 100
}

variable "disk_size" {
  type    = number
  default = 20
}
