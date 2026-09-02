# The Fuck

*The Fuck* is a magnificent app, inspired by a tweet by @liamosaur, that corrects errors in previous console commands.

Is *The Fuck* too slow? Try the experimental instant mode!

## Examples

When you try to run a command that fails because of permissions, *The Fuck* suggests the corrected command:

    apt-get install vim
    E: Could not open lock file /var/lib/dpkg/lock - open (13: Permission denied)
    E: Unable to lock the administration directory (/var/lib/dpkg/), are you root?
    
    fuck
    sudo apt-get install vim [enter/↑/↓/ctrl+c]

When you try to push to a branch that has no upstream, *The Fuck* adds the necessary flag:

    git push
    fatal: The current branch master has no upstream branch.
    
    fuck
    git push --set-upstream origin master [enter/↑/↓/ctrl+c]

When you misspell a command, *The Fuck* suggests the closest match:

    puthon
    No command 'puthon' found, did you mean:
     Command 'python' from package 'python-minimal' (main)
     Command 'python3' from package 'python3' (main)
    zsh: command not found: puthon
    
    fuck
    python [enter/↑/↓/ctrl+c]

And so on for misspelled git subcommands, lein tasks, and many other tools.

If you're not afraid of blindly running corrected commands, the require_confirmation settings option can be disabled.

## Requirements

- python (3.5+)
- pip
- python-dev

## Installation

On macOS or Linux, you can install *The Fuck* via Homebrew. On Ubuntu or Mint, use the apt package manager followed by pip. FreeBSD and ChromeOS are supported via their respective package managers. Arch-based systems can use pacman. On other systems, install via pip.

It is recommended that you place this command in your .bash_profile, .bashrc, .zshrc or other startup script:

    eval $(thefuck --alias)

You can use whatever you want as an alias, like for Mondays:

    eval $(thefuck --alias FUCK)

Changes are only available in a new shell session. To make changes immediately available, run source ~/.bashrc (or your shell config file like .zshrc).

To run fixed commands without confirmation, use the --yeah option:

    fuck --yeah

To fix commands recursively until succeeding, use the -r option:

    fuck -r

## Updating

    pip3 install thefuck --upgrade

## Uninstall

To remove *The Fuck*, reverse the installation process: erase or comment thefuck alias line from your shell config, and use your package manager to uninstall the binaries.

## How it works

*The Fuck* attempts to match the previous command with a rule. If a match is found, a new command is created using the matched rule and executed. The default rules cover a wide range of common mistakes across many tools: adb, ag, aws, cargo, cat, cd, chmod, composer, conda, cp, django, docker, fabric, gem, git, go, gradle, grep, grunt, gulp, heroku, hg, java, javac, lein, ln, ls, man, mkdir, mvn, npm, open, pip, php, port, prove, python, rails, react-native, rm, scm, sed, ssh, sudo, systemctl, terraform, test, tmux, tsuru, vagrant, whois, virtualenvwrapper, yarn, and more.

The following commands are bundled with *The Fuck*, but are not enabled by default: git_push_force, which adds --force-with-lease to a git push, and rm_root, which adds --no-preserve-root to rm -rf / command.

## Creating your own rules

To add your own rule, create a file named your-rule-name.py in ~/.config/thefuck/rules. The rule file must contain two functions:

    match(command: Command) -> bool
    get_new_command(command: Command) -> str | list[str]

Additionally, rules can contain optional functions:

    side_effect(old_command: Command, fixed_command: str) -> None

Rules can also contain the optional variables enabled_by_default, requires_output and priority. Command has three attributes: script, output and script_parts. Your rule should not change Command.

To access a rule's settings, import it with from thefuck.conf import settings. settings is a special object assembled from ~/.config/thefuck/settings.py, and values from environment variables.

## Settings

Several *The Fuck* parameters can be changed in the file $XDG_CONFIG_HOME/thefuck/settings.py: rules, exclude_rules, require_confirmation, wait_command, no_colors, priority, debug, history_limit, alter_history, wait_slow_command, slow_commands, num_close_matches, and excluded_search_path_prefixes.

The same options can be set via environment variables prefixed with THEFUCK_.

## Third-party packages with rules

If you'd like to make a specific set of non-public rules, but would still like to share them with others, create a package named thefuck_contrib_* with a rules module. *The Fuck* will find rules located in the rules module.

## Experimental instant mode

The default behavior of *The Fuck* requires time to re-run previous commands. When in instant mode, *The Fuck* saves time by logging output with script, then reading the log.

Currently, instant mode only supports Python 3 with bash or zsh. zsh's autocorrect function also needs to be disabled in order for thefuck to work properly.

To enable instant mode, add --enable-experimental-instant-mode to the alias initialization in .bashrc, .bash_profile or .zshrc.

## Developing

See CONTRIBUTING.md.

## License

Project License can be found in LICENSE.md.