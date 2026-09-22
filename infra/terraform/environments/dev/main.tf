data "yandex_compute_image" "ubuntu" {
  family = "ubuntu-2204-lts"
}

data "yandex_vpc_subnet" "default_a" {
  name = "default-ru-central1-a"
}

resource "yandex_compute_instance" "vm" {
  name        = var.vm_name
  platform_id = var.platform_id
  zone        = var.zone

  resources {
    cores         = var.cores
    memory        = var.memory
    core_fraction = var.core_fraction
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.ubuntu.id
      size     = var.disk_size
    }
  }

  network_interface {
    subnet_id = data.yandex_vpc_subnet.default_a.id
    nat       = true
  }

  metadata = {
    ssh-keys  = "ubuntu:${var.ssh_public_key}"
    user-data = <<-EOF
      #cloud-config
      package_update: true
      package_upgrade: false
      packages:
        - docker.io
        - docker-compose
        - rsync
        - python3
        - python3-pip
      runcmd:
        - systemctl enable docker
        - systemctl start docker
        - usermod -aG docker ubuntu
    EOF
  }

  scheduling_policy {
    preemptible = true
  }

  allow_stopping_for_update = true
}
