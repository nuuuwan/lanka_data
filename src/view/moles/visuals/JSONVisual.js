export default function JSONVisual({ datumSet }) {
  return <pre>{JSON.stringify(datumSet, null, 2)}</pre>;
}
