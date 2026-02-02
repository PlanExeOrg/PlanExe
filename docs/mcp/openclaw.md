# OpenClaw

Homepage: [openclaw.ai](https://openclaw.ai/)

I'm installing OpenClaw on my old 2015 MacBook Pro.

Specs:
- **OS** macOS Monterey, Version 12.7.6
- **Processor** 2,8 GHz Quad-Core Intel Core i7
- **Memory** 16 GB 1600 MHz DDR3
- **Graphics** Intel Iris Pro 1536 MB
- **Storage** 500 GB SSD

## Prepare the computer

- Erase disk
- Reinstall macOS
- Update to newest available macOS version
- Waiting for it to restart multiple times.

This took around 5 hours.

## Install homebrew

Follow the instructions on the [brew.sh](https://brew.sh/) website.

## Install Node

```bash
brew install node
This runs for ages
Error: An exception occurred within a child process:
  FormulaUnavailableError: No available formula with the name "formula.jws.json"
```

```bash
brew reinstall llm
This ran for 4 hours
```

```bash
brew install node
Error: You are using macOS 12.
We (and Apple) do not provide support for this old version.
You may have better luck with MacPorts which supports older versions of macOS:
  https://www.macports.org
```

At this point, I think I may have better luck get my rasperry pi working.

## Install OpenClaw via node

```bash
npm i -g openclaw
openclaw onboard
```
