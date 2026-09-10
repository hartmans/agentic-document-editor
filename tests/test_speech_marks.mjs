import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const source = readFileSync(new URL('../src/audio/speechMarks.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
});
const { chunkForPosition, markAtTime, timeAtPosition } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`
);

const marks = [
  { timeMs: 100, docCharStart: 50, docCharEnd: 55 },
  { timeMs: 700, docCharStart: 57, docCharEnd: 62 },
];

test('time and character mapping uses preceding word, including boundaries', () => {
  assert.equal(timeAtPosition(marks, 49), 0);
  assert.equal(timeAtPosition(marks, 50), 100);
  assert.equal(timeAtPosition(marks, 56), 100);
  assert.equal(timeAtPosition(marks, 57), 700);
  assert.equal(timeAtPosition(marks, 100), 700);
  assert.equal(markAtTime(marks, 0), marks[0]);
  assert.equal(markAtTime(marks, 699), marks[0]);
  assert.equal(markAtTime(marks, 700), marks[1]);
  assert.equal(markAtTime(marks, 10000), marks[1]);
  assert.equal(timeAtPosition([], 100), 0);
  assert.equal(markAtTime([], 100), undefined);
});

test('character resume targets select the containing chunk', () => {
  const chunks = [
    { chunk_id: 0, doc_char_start: 0, doc_char_end: 50 },
    { chunk_id: 1, doc_char_start: 50, doc_char_end: 100 },
  ];
  assert.equal(chunkForPosition(chunks, 49), 0);
  assert.equal(chunkForPosition(chunks, 50), 1);
  assert.equal(chunkForPosition(chunks, 100), 1);
  assert.equal(chunkForPosition([], 0), 0);
});
