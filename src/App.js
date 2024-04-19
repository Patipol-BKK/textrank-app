import React from 'react';
import logo from './logo.svg';
import './App.css';
import axios from 'axios';

import { Card, CardHeader, CardBody, CardFooter, Text ,
  Heading,
  Stack,
  StackDivider,
  Box,
  Container,
  FormControl,
  FormLabel,
  Input,
  HStack,
  Checkbox,
  Button,
  Divider,
  Link,
  Flex,
  Textarea
} from '@chakra-ui/react'

import {
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  SliderMark,
} from '@chakra-ui/react'


function App() {
    let [inputText, setInputText] = React.useState('')

    let handleInputChange = (e) => {
        let inputValue = e.target.value
        setInputText(inputValue)
    }

    let [keywordText, setKeywordText] = React.useState('')

    let handleSetKeywordText = (e) => {
        setKeywordText(e)
    }

    let [coOccurances, setCoOccurances] = React.useState(2)

    let handleCoOccurances = (e) => {
        setCoOccurances(e)
    }

    let [directed, setDirected] = React.useState(0)

    let handleSetDirected = (e) => {
        if (e.target.checked === 'true')
            setDirected(1)
        else
            setDirected(0)
    }

    let [maxKeyWord, setMaxKeyWord] = React.useState(10)

    let handleSetMaxKeyWord = (e) => {
        setMaxKeyWord(e)
    }

    let [maxKeyWordLength, setMaxKeyWordLength] = React.useState(3)

    let handleSetMaxKeyWordLength = (e) => {
        setMaxKeyWordLength(e)
    }

    const fetchData = async () => {
        try {
            const response = await axios.get(`http://localhost:5000/run/${coOccurances}/${directed}/${maxKeyWord}/${maxKeyWordLength}?text=${inputText.replace(/\n/g, " ")}`);
            setKeywordText(String(response.data).replace(/,/g, ", "))
            console.log(response.data)
        } catch (error) {
            console.error('Error fetching data:', error)
        }
    }

    const labelStyles = {
        mt: '2',
        ml: '-2.5',
        fontSize: 'sm',
    }
  return (
    <Flex
        flexDirection="column"
        width="100wh"
        height="100vh"
        backgroundColor="gray.200"
        justifyContent="center"
        alignItems="center"
    >
        <Card>
            <CardHeader>
                <Heading>Article Keyword Extractor</Heading>
            </CardHeader>
            <CardBody>
                <HStack align="flex-start">
                    <Stack>
                        <Heading as='h4' size='md' p='1'>Input Article Here</Heading>
                        <Textarea
                            value={inputText}
                            backgroundColor='white'
                            onChange={handleInputChange}
                            placeholder=''
                            size='lg'
                            resize='none'

                            minH='400px'
                            minW='800px'
                        />
                        <Heading as='h4' size='md' p='1'>Generated Keywords</Heading>
                        <Textarea
                            value={keywordText}
                            backgroundColor='white'
                            placeholder=''
                            size='lg'
                            resize='none'

                            minH='200px'
                            minW='800px'
                        />
                    </Stack>
                    <Stack alignItems='top-left' p='1' pl='5' minW='200px'>
                        <Heading as='h4' size='md' p=''>
                            Configs
                        </Heading>
                        <Text fontSize='md'>Co-Occurances: {coOccurances}</Text>
                        <Slider defaultValue={2} min={2} max={10} step={1} onChange={(val) => handleCoOccurances(val)}>
                            <SliderMark value={2} {...labelStyles}>
                                2
                            </SliderMark>
                            <SliderMark value={10} {...labelStyles}>
                                10
                            </SliderMark>
                            <SliderTrack bg='gray.200'>
                                <SliderFilledTrack bg='teal' />
                            </SliderTrack>
                            <SliderThumb boxSize={6} />
                        </Slider>

                        <Checkbox defaultChecked pt='8' onChange={(val) => handleSetDirected(val)}>Directed Graph</Checkbox>

                        <Text fontSize='md' pt='8'>Top Keyword: {maxKeyWord}</Text>
                        <Slider defaultValue={10} min={2} max={20} step={1} onChange={(val) => handleSetMaxKeyWord(val)}>
                            <SliderMark value={2} {...labelStyles}>
                                2
                            </SliderMark>
                            <SliderMark value={20} {...labelStyles}>
                                20
                            </SliderMark>
                            <SliderTrack bg='gray.200'>
                                <SliderFilledTrack bg='teal' />
                            </SliderTrack>
                            <SliderThumb boxSize={6} />
                        </Slider>

                        <Text fontSize='md' pt='8'>Max Keyword Length: {maxKeyWordLength}</Text>
                        <Slider defaultValue={2} min={0} max={10} step={1} onChange={(val) => handleSetMaxKeyWordLength(val)}>
                            <SliderMark value={0} {...labelStyles}>
                                0
                            </SliderMark>
                            <SliderMark value={10} {...labelStyles}>
                                10
                            </SliderMark>
                            <SliderTrack bg='gray.200'>
                                <SliderFilledTrack bg='teal' />
                            </SliderTrack>
                            <SliderThumb boxSize={6} />
                        </Slider>

                        <Button colorScheme='teal' onClick={fetchData} mt='8'>Generate Keywords!</Button>
                    </Stack>
                </HStack>
            </CardBody>
        </Card>
    </Flex>
  );
}

export default App;
