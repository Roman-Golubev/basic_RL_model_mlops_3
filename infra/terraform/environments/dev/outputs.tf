output "vm_external_ip" {
  value = yandex_compute_instance.vm.network_interface[0].nat_ip_address
}

output "vm_internal_ip" {
  value = yandex_compute_instance.vm.network_interface[0].ip_address
}

output "vm_id" {
  value = yandex_compute_instance.vm.id
}
