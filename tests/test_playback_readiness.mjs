import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const source = readFileSync(new URL('../src/audio/playbackReadiness.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
});
const { marksForStart } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`
);
const marks = [{ timeMs: 700, docCharStart: 57, docCharEnd: 62 }];

test('chunk starts and exact audio resumes proceed while marks are pending', async () => {
  const pending = new Promise(() => {});
  assert.deepEqual(await marksForStart(50, 50, undefined, pending, []), []);
  assert.equal(await marksForStart(57, 50, 700, pending, marks), marks);
});

test('a character seek waits for timing', async () => {
  let release;
  const pending = new Promise((resolve) => { release = resolve; });
  let ready = false;
  const result = marksForStart(57, 50, undefined, pending, []).then((value) => {
    ready = true;
    return value;
  });
  await Promise.resolve();
  assert.equal(ready, false);
  release({ marks, error: null });
  assert.equal(await result, marks);
});

test('alignment failure blocks character seeks but permits chunk starts', async () => {
  const failure = Promise.resolve({ marks: [], error: new Error('alignment failed') });
  assert.deepEqual(await marksForStart(50, 50, undefined, failure, []), []);
  await assert.rejects(marksForStart(57, 50, undefined, failure, []), /alignment failed/);
});
