// *****************************************************************************
// Copyright (C) 2026 EclipseSource GmbH.
//
// This program and the accompanying materials are made available under the
// terms of the Eclipse Public License v. 2.0 which is available at
// http://www.eclipse.org/legal/epl-2.0.
//
// This Source Code may also be made available under the following Secondary
// Licenses when the conditions for such availability set forth in the Eclipse
// Public License v. 2.0 are satisfied: GNU General Public License, version 2
// with the GNU Classpath Exception which is available at
// https://www.gnu.org/software/classpath/license.html.
//
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-only WITH Classpath-exception-2.0
// *****************************************************************************

import { ChatResponsePartRenderer } from '../chat-response-part-renderer';
import { injectable } from '@theia/core/shared/inversify';
import { AppliedContextEdit } from '@theia/ai-core';
import { ChatResponseContent, ContextEditChatResponseContent } from '@theia/ai-chat/lib/common';
import { ReactNode } from '@theia/core/shared/react';
import { nls } from '@theia/core/lib/common/nls';
import * as React from '@theia/core/shared/react';

@injectable()
export class ContextEditPartRenderer implements ChatResponsePartRenderer<ContextEditChatResponseContent> {

    canHandle(response: ChatResponseContent): number {
        if (ContextEditChatResponseContent.is(response)) {
            return 10;
        }
        return -1;
    }

    render(response: ContextEditChatResponseContent): ReactNode {
        return (
            <div className='theia-contextEdit'>
                {response.edits.map((edit, index) => <div key={index}>{this.describeEdit(edit)}</div>)}
            </div>
        );
    }

    protected describeEdit(edit: AppliedContextEdit): string {
        if (edit.cleared_thinking_turns !== undefined) {
            return nls.localize(
                'theia/ai/chat-ui/contextEditClearedThinking',
                'Context edited: cleared thinking of {0} turn(s) ({1} tokens)',
                edit.cleared_thinking_turns, edit.cleared_input_tokens ?? 0);
        }
        if (edit.cleared_tool_uses !== undefined) {
            return nls.localize(
                'theia/ai/chat-ui/contextEditClearedToolUses',
                'Context edited: cleared {0} tool use(s) ({1} tokens)',
                edit.cleared_tool_uses, edit.cleared_input_tokens ?? 0);
        }
        return nls.localize('theia/ai/chat-ui/contextEditApplied', 'Context edited: {0}', edit.type);
    }
}
