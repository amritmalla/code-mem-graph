import { expect, test } from 'vitest'
import { PriceCard } from './PriceCard'

test('PriceCard displays the supplied cents', () => {
  expect(PriceCard({ cents: 123 }).props.children).toEqual([123, ' cents'])
})
