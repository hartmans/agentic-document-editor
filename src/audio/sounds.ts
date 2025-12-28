import instructionsSentUrl from "../asets/sounds/instructions_sent.ogg";
export {instructionsSentUrl};

export function play(url:string): Promise<void>
{
    const audio = new Audio(url);
    return audio.play();
}
