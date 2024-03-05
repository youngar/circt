import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';
import * as vscode from 'vscode';

import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind, RevealOutputChannelOn

} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
    const outputChannel = vscode.window.createOutputChannel('HL-LSPD');
    context.subscriptions.push(outputChannel);

    // Get the path to the language server.
    let serverOptions: ServerOptions = {
        command: '/Users/andrewyoung/wsp/circt/build-release/bin/hwml-lsp-server'
    };

    // Options to control the language client
    let clientOptions: LanguageClientOptions = {
        // Register the server for plain text documents
        documentSelector: [{ scheme: 'file', language: 'hl' }],
        synchronize: {
            // Notify the server about file changes to '.clientrc files contained in the workspace
            fileEvents: workspace.createFileSystemWatcher('**/.hl')
        },
        outputChannel: outputChannel,
        revealOutputChannelOn: RevealOutputChannelOn.Info,
    };

    // Create the language client and start the client.
    client = new LanguageClient(
        'HWML',
        'HWML Language Server',
        serverOptions,
        clientOptions
    );
    
    // The command has been defined in the package.json file
    // Now provide the implementation of the command with registerCommand
    // The commandId parameter must match the command field in package.json
    let disposable = vscode.commands.registerCommand('hl.helloworld', () => {
        // The code you place here will be executed every time your command is executed

        // Display a message box to the user
        vscode.window.showInformationMessage('Hello World!');
    });
    context.subscriptions.push(disposable);

    let restartCommand = vscode.commands.registerCommand('hl.restart', () => {
        // The code you place here will be executed every time your command is executed

        // Display a message box to the user
        vscode.window.showInformationMessage('Restart!');
    });
    context.subscriptions.push(restartCommand);

    outputChannel.show();
    outputChannel.appendLine("qwertyuiop");
    vscode.window.showInformationMessage("hello from the client!", "yes", "no");
    vscode.window.showErrorMessage("hello");

    // Start the client. This will also launch the server
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
