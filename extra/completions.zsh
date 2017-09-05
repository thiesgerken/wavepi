#compdef wavepi
_wavepi() {
  local -a commands
  commands=(
    '--help'
    '--version'
    '--export-config:generate config file and write to stdout (or --config [file])'
    '--config:config file input/output'
    '--config-format:export format (Text|LaTeX|Description|XML|JSON|ShortText)'
  )

  if (( CURRENT == 2 )); then
    _describe -t commands 'commands' commands
  fi

  return 0
}

_wavepi
