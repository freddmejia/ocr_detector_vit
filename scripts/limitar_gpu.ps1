# Limita la GPU NVIDIA del anfitrión para entrenar sin llevarla al máximo.
#
# Docker no puede limitar la GPU: el límite se aplica en Windows con nvidia-smi y afecta a todo el equipo.
# - Frecuencia máxima: PORCENTAJE % de la máxima de la GPU (70 % de 3090 MHz = 2163 MHz en la RTX 5060 Ti).
#   Es lo que más reduce consumo y temperatura.
# - Consumo máximo: el mínimo que admite la tarjeta (150 W en la RTX 5060 Ti, de 180 W por defecto).
# Ambos ajustes se pierden al reiniciar Windows.
#
# Uso (pide permisos de administrador):
#   powershell -ExecutionPolicy Bypass -File scripts\limitar_gpu.ps1              # limitar al 70 %
#   powershell -ExecutionPolicy Bypass -File scripts\limitar_gpu.ps1 -Porcentaje 60
#   powershell -ExecutionPolicy Bypass -File scripts\limitar_gpu.ps1 -Quitar      # volver a los valores de fábrica

param(
    [ValidateRange(30, 100)]
    [int]$Porcentaje = 70,
    [switch]$Quitar
)

$esAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $esAdmin) {
    $argumentos = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-NoExit", "-File", "`"$PSCommandPath`"")
    if ($Quitar) { $argumentos += "-Quitar" } else { $argumentos += @("-Porcentaje", $Porcentaje) }
    Start-Process powershell -Verb RunAs -ArgumentList $argumentos
    exit
}

function Consultar([string]$campo) {
    [double](nvidia-smi --query-gpu=$campo --format=csv,noheader,nounits).Trim()
}

if ($Quitar) {
    nvidia-smi -rgc
    nvidia-smi -pl (Consultar "power.default_limit")
} else {
    $frecuencia = [math]::Floor((Consultar "clocks.max.graphics") * $Porcentaje / 100)
    nvidia-smi -lgc "0,$frecuencia"
    nvidia-smi -pl (Consultar "power.min_limit")
}

Write-Host ""
nvidia-smi --query-gpu=name,clocks.max.graphics,power.limit,power.default_limit,temperature.gpu --format=csv
Write-Host "Para comprobar la frecuencia durante el entrenamiento: nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu --format=csv -l 5"
